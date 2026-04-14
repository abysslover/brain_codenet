import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AssociativeMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_size = config.memory_size
        self.feature_dim = config.snn_hidden_dim
        self.top_k = config.top_k_memories

        self.memory_keys = nn.Parameter(
            torch.randn(self.memory_size, self.feature_dim) * 0.1
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.memory_size, self.feature_dim) * 0.1
        )

        self.temperature = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.feature_dim)))

        self.layer_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, query: torch.Tensor):
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.memory_keys, dim=-1)

        similarity = torch.matmul(query_norm, keys_norm.t())
        similarity = similarity * self.temperature.abs()

        top_scores, top_indices = torch.topk(similarity, self.top_k, dim=-1)
        attention_weights = F.softmax(top_scores, dim=-1)

        selected_memories = self.memory_values[top_indices]
        retrieved = torch.sum(
            attention_weights.unsqueeze(-1) * selected_memories, dim=1
        )

        retrieved = self.layer_norm(retrieved + query)

        return retrieved, attention_weights
