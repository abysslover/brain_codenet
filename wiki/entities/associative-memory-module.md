---
title: AssociativeMemory
created: 2026-04-15
updated: 2026-04-15
tags: [entity, memory, hippocampus]
sources: []
---

# AssociativeMemory

The AssociativeMemory module emulates the hippocampus, performing content-addressable memory retrieval using a Modern Hopfield Network.

## Role

Retrieves relevant stored patterns based on similarity to the current input, enabling associative recall.

## Architecture

```
Query Vector → Similarity Computation → Top-k Selection → Weighted Retrieval → Retrieved State
                (Cosine Similarity)      (Sparse)         (Attention)
```

## Implementation

```python
class AssociativeMemory(nn.Module):
    def __init__(self, config):
        self.memory_size = 512
        self.feature_dim = config.snn_hidden_dim
        self.top_k = 8
        
        # Learnable memory (colocated keys and values)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, feature_dim) * 0.1)
        self.memory_values = nn.Parameter(torch.randn(memory_size, feature_dim) * 0.1)
        
        # Temperature for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0 / sqrt(feature_dim)))
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, query):
        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.memory_keys, dim=-1)
        
        # Compute similarities
        similarity = torch.matmul(query_norm, keys_norm.t())
        similarity = similarity * self.temperature.abs()
        
        # Sparse top-k retrieval
        top_scores, top_indices = torch.topk(similarity, self.top_k, dim=-1)
        attention_weights = F.softmax(top_scores, dim=-1)
        
        # Weighted combination
        selected_memories = self.memory_values[top_indices]
        retrieved = torch.sum(
            attention_weights.unsqueeze(-1) * selected_memories, 
            dim=1
        )
        
        # Residual connection
        retrieved = self.layer_norm(retrieved + query)
        
        return retrieved, attention_weights
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| memory_size | 512 | Number of stored patterns |
| feature_dim | 512 | Dimension of memory vectors |
| top_k | 8 | Number of memories to retrieve |
| temperature | learnable | Controls attention sharpness |

## Properties

1. **Sparse**: Only top-k memories activate
2. **Colocated**: Memory and computation unified
3. **Differentiable**: Full gradient support
4. **Content-Addressable**: Retrieve by similarity

## Related Entities

- [[wiki/entities/brain-coding-model|BrainCodingModel]]
- [[wiki/entities/spiking-encoder|SpikingEncoder]]
- [[wiki/entities/spiking-decoder|SpikingDecoder]]

## Related Concepts

- [[wiki/concepts/associative-memory|Associative Memory]]
- [[wiki/concepts/colocation|Colocation]]
- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
