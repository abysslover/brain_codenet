---
title: Colocation
created: 2026-04-15
updated: 2026-04-15
tags: [principle, memory, architecture]
sources: []
---

# Colocation

Colocation is a fundamental principle in brain-inspired computing where memory storage and computation occur in the same physical space.

## Overview

In traditional Von Neumann architecture, memory (RAM) and processing units (CPU/GPU) are physically separated. This creates the **Memory Wall** problem:

- Data must be fetched from memory to compute units
- Each fetch consumes significant energy
- In LLM autoregressive decoding, weights are repeatedly fetch-compute-discard

## Brain Approach

Biological neurons solve this through **colocation**:

- Synaptic weights (memory) are physically at the neuron (computation site)
- No data movement between separate memory and compute units
- Computation happens directly where data is stored

## Implementation

In this system, colocation is implemented in the `AssociativeMemory` module:

```python
class AssociativeMemory(nn.Module):
    # Memory keys and values are learned parameters (colocated)
    self.memory_keys = nn.Parameter(...)
    self.memory_values = nn.Parameter(...)
    
    def forward(self, query):
        # Similarity computation AND memory retrieval happen together
        # No separate fetch phase
        similarity = torch.matmul(query, keys.t())
        retrieved = weighted_sum(selected_values)
```

## Benefits

1. **Energy Efficiency**: Eliminates memory fetch costs
2. **Speed**: Computation and retrieval are unified
3. **Scalability**: No memory bandwidth bottleneck

## Related Concepts

- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/concepts/associative-memory|Associative Memory]]
- [[wiki/concepts/energy-efficiency|Energy Efficiency]]
