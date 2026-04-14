import time
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from tqdm import tqdm


class BrainTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 디바이스 설정
        self.device = config.resolve_device()
        print(f"[Trainer] Device: {self.device}")

        # CUDA 정보 출력
        if self.device.type == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"[Trainer] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # Multi-GPU 설정
        self.model = model.to(self.device)
        if config.multi_gpu and torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
            print(f"[Trainer] DataParallel: {torch.cuda.device_count()} GPUs")

        # AMP Scaler (CUDA 전용)
        self.use_amp = config.use_amp and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        print(f"[Trainer] AMP: {self.use_amp}")

        # 옵티마이저 & 스케줄러
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)

        # 체크포인트 경로
        self.ckpt_dir = Path(config.get_ckpt_dir())
        self.results_dir = Path(config.get_results_dir())

        # Best 모델 추적
        self.best_metric_value = (
            float("-inf") if config.best_metric_mode == "max" else float("inf")
        )
        self.start_epoch = 0

        # Resume 처리
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"[Trainer] Parameters: {total_params:,}")

        # 실험 히스토리 추적
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "top1_accuracy": [],
            "top5_accuracy": [],
            "avg_sparsity": [],
            "energy_reduction": [],
        }

    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _save_checkpoint(self, epoch: int, val_stats: dict, is_best: bool):
        """체크포인트 저장 - 모델, 옵티마이저, 스케줄러 상태 포함"""
        raw_model = (
            self.model.module if isinstance(self.model, DataParallel) else self.model
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric_value": self.best_metric_value,
            "val_stats": val_stats,
            "config": self.config.__dict__,
        }

        # 매 epoch 저장
        if self.config.save_every_epoch and not self.config.save_best_only:
            epoch_path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(payload, epoch_path)

        # Best 모델 저장
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(payload, best_path)
            print(f"  [CKPT] Best model updated: {best_path}")

        # Latest 항상 저장 (재개용)
        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(payload, latest_path)

    def _load_checkpoint(self, ckpt_path: str):
        """체크포인트에서 학습 재개"""
        if not Path(ckpt_path).exists():
            print(f"[CKPT] 경로 없음: {ckpt_path}")
            return

        print(f"[CKPT] Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        raw_model = (
            self.model.module if isinstance(self.model, DataParallel) else self.model
        )
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric_value = ckpt.get("best_metric_value", self.best_metric_value)

        print(f"[CKPT] Resumed from epoch {ckpt['epoch']}")

    def _is_best(self, val_stats: dict) -> bool:
        """Best 모델 판단"""
        current = val_stats.get(self.config.best_metric, 0.0)
        if self.config.best_metric_mode == "max":
            if current > self.best_metric_value:
                self.best_metric_value = current
                return True
        else:
            if current < self.best_metric_value:
                self.best_metric_value = current
                return True
        return False

    def train_epoch(self, epoch: int) -> dict:
        """AMP 적용된 학습 epoch"""
        self.model.train()

        total_loss = 0.0
        total_sparsity = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # AMP autocast
            with autocast(enabled=self.use_amp):
                output = self.model(input_ids, attention_mask, labels)
                loss = output["loss"]

            # Scaled backward
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale 후 적용)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_sparsity += output["encoder_stats"]["sparsity"]
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "sparsity": f"{output['encoder_stats']['sparsity']:.3f}",
                    "amp": str(self.use_amp),
                }
            )

        self.scheduler.step()

        return {
            "avg_loss": total_loss / max(num_batches, 1),
            "avg_sparsity": total_sparsity / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader, desc="Val") -> dict:
        """검증 루프 - AMP 적용"""
        self.model.eval()

        total_loss = 0.0
        total_sparsity = 0.0
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        num_batches = 0

        pbar = tqdm(loader, desc=desc)

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                output = self.model(input_ids, attention_mask, labels)

            logits = output["logits"]
            target = labels[:, 0]
            valid_mask = target != self.config.pad_token_id

            if valid_mask.sum() == 0:
                continue

            logits_valid = logits[valid_mask]
            target_valid = target[valid_mask]

            # Top-1 Accuracy
            pred_top1 = logits_valid.argmax(dim=-1)
            top1_correct += (pred_top1 == target_valid).sum().item()

            # Top-5 Accuracy
            pred_top5 = logits_valid.topk(5, dim=-1).indices
            top5_correct += (
                (pred_top5 == target_valid.unsqueeze(-1)).any(dim=-1).sum().item()
            )

            total_samples += valid_mask.sum().item()
            total_loss += output["task_loss"].item()
            total_sparsity += output["encoder_stats"]["sparsity"]
            num_batches += 1

        avg_sparsity = total_sparsity / max(num_batches, 1)

        return {
            "loss": total_loss / max(num_batches, 1),
            "top1_accuracy": top1_correct / max(total_samples, 1),
            "top5_accuracy": top5_correct / max(total_samples, 1),
            "avg_sparsity": avg_sparsity,
            "energy_reduction": 1.0 / max(1.0 - avg_sparsity, 1e-6),
        }

    def save_results(self, epoch, train_stats, val_stats):
        """매 epoch 결과를 JSON 과 CSV 에 저장"""
        self.history["train_loss"].append(train_stats["avg_loss"])
        self.history["val_loss"].append(val_stats["loss"])
        self.history["top1_accuracy"].append(val_stats["top1_accuracy"])
        self.history["top5_accuracy"].append(val_stats["top5_accuracy"])
        self.history["avg_sparsity"].append(val_stats["avg_sparsity"])
        self.history["energy_reduction"].append(val_stats["energy_reduction"])

        # JSON 저장
        with open(self.results_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # CSV 저장 (논문 테이블용)
        with open(self.results_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history.keys())
            writer.writeheader()
            for i in range(len(self.history["train_loss"])):
                writer.writerow({k: v[i] for k, v in self.history.items()})

    def train(self):
        """전체 학습 루프 - 체크포인트 저장 포함"""
        print(f"\n{'=' * 60}")
        print(f"  BrainCodeNet Training: {self.config.experiment_name}")
        print(f"  Device: {self.device} | AMP: {self.use_amp}")
        print(f"  Start from epoch: {self.start_epoch}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            t0 = time.time()

            # 학습
            train_stats = self.train_epoch(epoch)

            # 검증
            val_stats = self.evaluate(self.val_loader, desc=f"Val Epoch {epoch + 1}")

            elapsed = time.time() - t0

            # Best 판단 후 체크포인트 저장
            is_best = self._is_best(val_stats)
            self._save_checkpoint(epoch, val_stats, is_best)

            # CUDA 메모리 상태 (선택적)
            if self.device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                vram_info = f"VRAM {alloc:.1f}/{reserved:.1f} GB"
            else:
                vram_info = "N/A"

            print(f"\n[Epoch {epoch + 1:03d}/{self.config.num_epochs}]")
            print(f"  Time: {elapsed:.1f}s | {vram_info}")
            print(f"  Train Loss: {train_stats['avg_loss']:.4f}")
            print(
                f"  Val Loss: {val_stats['loss']:.4f} | Top-1: {val_stats['top1_accuracy']:.4f}"
            )
            print(f"  Val Top-5: {val_stats['top5_accuracy']:.4f}")
            print(
                f"  Sparsity: {val_stats['avg_sparsity']:.4f} | Energy: {val_stats['energy_reduction']:.2f}x"
            )

            if is_best:
                print(
                    f"  ★ Best {self.config.best_metric}: {self.best_metric_value:.4f}"
                )

        print(f"\n{'=' * 60}")
        print(f"  Training Complete | Best: {self.best_metric_value:.4f}")
        print(f"{'=' * 60}\n")
