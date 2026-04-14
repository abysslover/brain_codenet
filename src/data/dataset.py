import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
from config import BrainCodingConfig


class CodingQADataset(Dataset):
    """
    PyTorch Dataset for HuggingFace coding instruction datasets.
    Combines instruction and input to create questions, with output as answers.
    """

    def __init__(
        self, tokenizer: AutoTokenizer, config: BrainCodingConfig, split: str = "train"
    ):
        self.dataset = load_dataset(
            config.dataset_name, split=f"{split}[:{config.max_samples}]"
        )
        self.tokenizer = tokenizer
        self.config = config

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        if input_text.strip():
            question = f"Instruction: {instruction}\nInput: {input_text}"
        else:
            question = instruction

        answer = item.get("output", "")

        q_tokens = self.tokenizer(
            question,
            max_length=self.config.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        a_tokens = self.tokenizer(
            answer,
            max_length=self.config.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": q_tokens["input_ids"].squeeze(0),
            "attention_mask": q_tokens["attention_mask"].squeeze(0),
            "labels": a_tokens["input_ids"].squeeze(0),
        }


def create_dataloaders(config: BrainCodingConfig):
    """
    Create optimized training and validation DataLoaders.

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        tokenizer: GPT-2 tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = config.resolve_device()

    # CUDA 환경에서만 pin_memory 활성화
    use_pin_memory = config.pin_memory and (device.type == "cuda")

    # num_workers 가 0 이면 prefetch_factor 무시
    prefetch = config.prefetch_factor if config.num_workers > 0 else None

    full_dataset = CodingQADataset(tokenizer, config, "train")

    # Train / Val 분리
    total = len(full_dataset)
    val_size = max(1, int(total * config.val_split_ratio))
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # 최적화된 DataLoader 설정
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=(config.num_workers > 0),
        drop_last=False,
    )

    print(f"[DataLoader] Device: {device}")
    print(f"[DataLoader] Train: {train_size}, Val: {val_size}")
    print(f"[DataLoader] pin_memory: {use_pin_memory}, workers: {config.num_workers}")

    return train_loader, val_loader, tokenizer
