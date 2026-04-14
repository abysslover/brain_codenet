import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBaselineModel(nn.Module):
    """
    BrainCodeNet 과 공정한 비교를 위한 Dense Baseline
    동일한 파라미터 수를 가지는 표준 Transformer 구조
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 동일한 임베딩 차원
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Transformer Encoder (SNN 대체)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=4,
            dim_feedforward=config.snn_hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Dense Projection (AssociativeMemory 대체)
        self.memory_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.snn_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.snn_hidden_dim),
            nn.Dropout(0.1),
        )

        # Output Head (SpikingDecoder 대체)
        self.output_head = nn.Sequential(
            nn.Linear(config.snn_hidden_dim, config.snn_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.snn_hidden_dim // 2, config.vocab_size),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # 임베딩
        x = self.embedding(input_ids)  # [B, L, D]

        # Transformer 인코딩
        padding_mask = attention_mask == 0
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling (attention mask 적용)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # Dense 연산
        h = self.memory_proj(x)
        logits = self.output_head(h)

        output = {
            "logits": logits,
            "encoder_stats": {
                "sparsity": 0.0,  # Dense 모델은 희소성 없음
                "firing_rate": 1.0,
                "total_spikes": float(logits.numel()),
            },
        }

        if labels is not None:
            target = labels[:, 0]
            task_loss = F.cross_entropy(
                logits, target, ignore_index=self.config.pad_token_id
            )
            output.update(
                {
                    "loss": task_loss,
                    "task_loss": task_loss,
                    "energy_loss": torch.tensor(0.0, device=logits.device),
                }
            )

        return output
