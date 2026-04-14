import copy
import json
import time
import torch
from pathlib import Path
from typing import List, Dict

from config import BrainCodingConfig
from data.dataset import create_dataloaders
from models.brain_model import BrainCodingModel
from models.baseline_model import DenseBaselineModel
from training.trainer import BrainTrainer


class ExperimentRunner:
    """
    Dense Baseline, BrainCodeNet, Ablation Study 를 자동으로 실행하고
    결과를 수집하여 LaTeX 테이블 생성을 위한 데이터를 준비하는 클래스
    """

    def __init__(self, base_config: BrainCodingConfig):
        self.base_config = base_config
        self.results = []

        # 결과 저장 경로
        self.wiki_dir = Path(base_config.get_wiki_sources_dir())
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        """재현성을 위한 시드 설정"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_single_experiment(
        self, config: BrainCodingConfig, model_class, label: str
    ) -> Dict:
        """
        단일 실험 실행 및 최종 검증 결과 반환

        Args:
            config: 실험 설정
            model_class: 사용할 모델 클래스 (BrainCodingModel 또는 DenseBaselineModel)
            label: 테이블에 표시될 모델명

        Returns:
            dict: 최종 검증 메트릭
        """
        print(f"\n{'=' * 60}")
        print(f"  실험 시작: {label}")
        print(f"{'=' * 60}")

        self._set_seed(config.seed)

        # 데이터 로더 생성
        train_loader, val_loader, tokenizer = create_dataloaders(config)

        # 모델 초기화
        model = model_class(config)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  모델 파라미터: {total_params:,}")

        # 학습 실행
        trainer = BrainTrainer(model, train_loader, val_loader, config)
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # 최종 평가 (best checkpoint 로드)
        best_ckpt = Path(config.get_ckpt_dir()) / "best_model.pt"
        if best_ckpt.exists():
            trainer._load_checkpoint(str(best_ckpt))

        final_stats = trainer.evaluate(val_loader, desc=f"Final Eval [{label}]")

        # 결과 정리
        result = {
            "model": label,
            "experiment_name": config.experiment_name,
            "total_params": total_params,
            "training_time": training_time,
            "top1_accuracy": final_stats["top1_accuracy"],
            "avg_sparsity": final_stats["avg_sparsity"],
            "energy_reduction": final_stats["energy_reduction"],
            "val_loss": final_stats["loss"],
        }

        self.results.append(result)
        self._save_intermediate_results()

        print(
            f"  완료: Top-1 {result['top1_accuracy']:.3f} | "
            f"Sparsity {result['avg_sparsity']:.3f} | "
            f"Energy {result['energy_reduction']:.1f}x"
        )

        return result

    def _save_intermediate_results(self):
        """실험 중간 결과 저장 (실험 중단 대비)"""
        with open(self.wiki_dir / "intermediate_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

    def run_baseline_comparison(self) -> List[Dict]:
        """Dense Baseline vs BrainCodeNet 비교 실험"""
        print("\n[Phase 1] Baseline Comparison")

        # 1. Dense Baseline
        baseline_config = copy.deepcopy(self.base_config)
        baseline_config.experiment_name = "dense_baseline"
        self.run_single_experiment(
            baseline_config, DenseBaselineModel, "Dense Baseline"
        )

        # 2. BrainCodeNet (Full)
        brain_config = copy.deepcopy(self.base_config)
        brain_config.experiment_name = "braincodeNet_full"
        self.run_single_experiment(brain_config, BrainCodingModel, "BrainCodeNet")

        return self.results[-2:]  # 마지막 2 개 결과 반환

    def run_ablation_study(self) -> List[Dict]:
        """Ablation Study 자동 실행"""
        print("\n[Phase 2] Ablation Study")

        variants = self.base_config.get_ablation_variants()
        ablation_results = []

        for label, config in variants.items():
            if label == "BrainCodeNet":
                continue  # 이미 baseline comparison 에서 실행됨

            result = self.run_single_experiment(config, BrainCodingModel, label)
            ablation_results.append(result)

        return ablation_results

    def run_all_experiments(self) -> List[Dict]:
        """전체 실험 파이프라인 실행"""
        print("BrainCodeNet 전체 실험 파이프라인 시작")

        # Baseline 비교
        self.run_baseline_comparison()

        # Ablation Study
        self.run_ablation_study()

        # 최종 결과 저장
        final_path = self.wiki_dir / "all_results.json"
        with open(final_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n전체 실험 완료. 결과 저장: {final_path}")
        return self.results
