---
title: SpikingEncoder
created: 2026-04-15
updated: 2026-04-15
tags: [entity, encoder, sensory]
sources: []
---

# SpikingEncoder

The SpikingEncoder emulates the sensory cortex, converting text tokens into temporal spike patterns.

## Role

Transforms input text into sparse, time-varying spike trains that can be processed by subsequent SNN layers.

## Architecture

```
Input Tokens → Embedding → Rate Coding → LIF Layer 1 → LIF Layer 2 → Encoded State
                                    (Temporal Processing)
```

## Implementation

```python
class SpikingEncoder(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Layer 1: Embedding to hidden
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=surrogate)
        
        # Layer 2: Hidden to hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=surrogate)
    
    def forward(self, input_ids, attention_mask):
        # Embedding
        x = self.embedding(input_ids)
        
        # Rate coding: convert to spike trains
        spike_trains = rate_encoding(x, num_time_steps)
        
        # Temporal processing through LIF layers
        for t in range(num_time_steps):
            x_t = spike_trains[t].mean(dim=1)  # Pool across sequence
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
        
        # Aggregate spike history
        encoded_state = spk2_history.mean(dim=0)
        
        return encoded_state, energy_stats
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| vocab_size | 50257 | GPT-2 vocabulary |
| embed_dim | 256 | Embedding dimension |
| hidden_dim | 512 | LIF layer dimension |
| num_time_steps | 20 | Temporal processing steps |
| beta | 0.9 | Membrane decay |
| threshold | 1.0 | Spike threshold |

## Output

- `encoded_state`: [batch, hidden_dim] - Final neural representation
- `energy_stats`: Dict with sparsity, firing_rate, total_spikes

## Related Entities

- [[wiki/entities/brain-coding-model|BrainCodingModel]]
- [[wiki/entities/associative-memory-module|AssociativeMemory]]

## Related Concepts

- [[wiki/concepts/leaky-integrate-and-fire|Leaky Integrate-and-Fire]]
- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/concepts/surrogate-gradient|Surrogate Gradient]]
