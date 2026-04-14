import os
import torch
from dataclasses import dataclass
from typing import Dict, Optional
import copy


@dataclass
class BrainCodingConfig:
    """
    Brain emulation-based Q&A system configuration.
    Manages all hyperparameters for a coding Q&A system based on brain efficiency
    principles (colocation, sparse spiking, event-driven processing).
    """

    # ========================================================================
    # Dataset Configuration
    # ========================================================================
    dataset_name: str = "iamtarun/python_code_instructions_18k_alpaca"
    max_samples: int = 1000  # For initial experiments
    max_input_length: int = 128
    max_output_length: int = 64
    val_split_ratio: float = 0.2

    # ========================================================================
    # SNN Parameters (Emulating Brain Dynamics)
    # ========================================================================
    vocab_size: int = 50257
    embed_dim: int = 256
    snn_hidden_dim: int = 512
    num_time_steps: int = 20
    beta: float = 0.9  # membrane decay factor
    threshold: float = 1.0  # spike firing threshold
    spike_grad_slope: float = 25.0  # surrogate gradient slope

    # ========================================================================
    # Associative Memory Configuration (Hippocampus Emulation)
    # ========================================================================
    memory_size: int = 512
    top_k_memories: int = 8

    # ========================================================================
    # Training Parameters
    # ========================================================================
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 15
    energy_loss_weight: float = 0.01
    sparsity_target: float = 0.1

    # ========================================================================
    # Ablation Study Control Flags
    # ========================================================================
    use_associative_memory: bool = True
    use_energy_regularizer: bool = True

    # ========================================================================
    # CUDA Optimization Settings
    # ========================================================================
    device: str = "cuda"
    use_amp: bool = True  # Automatic Mixed Precision
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    multi_gpu: bool = False  # DataParallel 사용 여부

    # ========================================================================
    # Checkpoint Settings
    # ========================================================================
    ckpt_dir: str = "checkpoints"
    save_every_epoch: bool = True
    save_best_only: bool = False  # True 면 best 만, False 면 매 epoch 저장
    best_metric: str = "top1_accuracy"
    best_metric_mode: str = "max"  # max 또는 min
    resume_from: Optional[str] = None  # 재개할 ckpt 경로

    # ========================================================================
    # Experiment Management
    # ========================================================================
    experiment_name: str = "braincodeNet_default"
    results_dir: str = "results"
    wiki_sources_dir: str = "wiki/sources"  # 논문용 출력 경로
    seed: int = 42
    pad_token_id: int = 0

    def resolve_device(self) -> torch.device:
        """디바이스 자동 감지 및 설정"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_ckpt_dir(self) -> str:
        """체크포인트 디렉토리 생성 및 반환"""
        path = os.path.join(self.ckpt_dir, self.experiment_name)
        os.makedirs(path, exist_ok=True)
        return path

    def get_results_dir(self) -> str:
        """결과 디렉토리 생성 및 반환"""
        path = os.path.join(self.results_dir, self.experiment_name)
        os.makedirs(path, exist_ok=True)
        return path

    def get_wiki_sources_dir(self) -> str:
        """wiki/sources 디렉토리 생성 및 반환"""
        os.makedirs(self.wiki_sources_dir, exist_ok=True)
        return self.wiki_sources_dir

    def get_latex_table_path(self) -> str:
        """LaTeX 테이블 파일 경로 반환"""
        return os.path.join(
            self.get_wiki_sources_dir(), "brain_codenet_result_table.tex"
        )

    def get_ablation_variants(self) -> Dict[str, "BrainCodingConfig"]:
        """Ablation Study 용 설정 변형 자동 생성"""
        variants = {}

        # 1. BrainCodeNet (제안 모델)
        base = copy.deepcopy(self)
        base.experiment_name = "BrainCodeNet"
        variants["BrainCodeNet"] = base

        # 2. SNN without Memory
        no_memory = copy.deepcopy(self)
        no_memory.use_associative_memory = False
        no_memory.experiment_name = "SNN_NoMemory"
        variants["SNN (No Memory)"] = no_memory

        # 3. SNN without Energy Regularizer
        no_energy = copy.deepcopy(self)
        no_energy.use_energy_regularizer = False
        no_energy.energy_loss_weight = 0.0
        no_energy.experiment_name = "SNN_NoEnergyReg"
        variants["SNN (No Energy Reg)"] = no_energy

        return variants
