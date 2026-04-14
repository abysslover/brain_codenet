---
title: Associative Memory
created: 2026-04-15
updated: 2026-04-15
tags: [memory, hippocampus, retrieval]
sources: []
---

# Associative Memory

Associative memory is a content-addressable memory system that retrieves information based on similarity to a query, emulating the brain's hippocampus.

## Overview

### Traditional Memory (Von Neumann)

```
Address-based: memory[0x1234] = value
                value = memory[0x1234]  # Need exact address
```

### Associative Memory (Brain-Inspired)

```
Content-based: memory = [pattern1, pattern2, pattern3]
                query = similar_to(pattern2)
                result = retrieve(query)  # Returns pattern2
```

## Modern Hopfield Network

The implementation uses Modern Hopfield Networks with energy-based retrieval:

### Energy Function

```
E(x, m_i) = -β·x·m_i^T  # Lower energy = better match
```

### Retrieval Dynamics

```
α_i = exp(β·x·m_i^T) / Σ_j exp(β·x·m_j^T)  # Softmax attention
retrieved = Σ_i α_i · m_i  # Weighted combination
```

## Implementation

```python
class AssociativeMemory(nn.Module):
    def __init__(self, memory_size, feature_dim, top_k=8):
        self.memory_keys = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.top_k = top_k
    
    def forward(self, query):
        # Cosine similarity
        similarity = torch.matmul(query, keys.t())
        
        # Sparse top-k retrieval
        top_scores, top_indices = torch.topk(similarity, self.top_k)
        
        # Weighted combination
        retrieved = sum(softmax(top_scores) * selected_values)
        
        return retrieved, attention_weights
```

## Key Properties

1. **Content-Addressable**: Retrieve by similarity, not address
2. **Sparse**: Only top-k memories activate
3. **Robust**: Handles noisy or incomplete queries
4. **Colocated**: Memory and computation unified

## Biological Inspiration

The hippocampus performs:

- Pattern completion (partial → complete memory)
- Pattern separation (similar inputs → distinct memories)
- Rapid one-shot learning

## Related Concepts

- [[wiki/concepts/colocation|Colocation]]
- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/entities/associative-memory-module|AssociativeMemory]]
- [[wiki/syntheses/architecture-overview|System Architecture]]
