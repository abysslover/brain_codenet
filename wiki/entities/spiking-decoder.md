---
title: SpikingDecoder
created: 2026-04-15
updated: 2026-04-15
tags: [entity, decoder, motor]
sources: []
---

# SpikingDecoder

The SpikingDecoder emulates the motor cortex, converting neural representations back into code tokens.

## Role

Transforms the processed spike patterns from the memory module into output token predictions.

## Architecture

```
Encoded State → LIF Processing → Spike History → Average → Output Projection → Token Logits
               (Temporal)        (Aggregation)            (Vocabulary)
```

## Implementation

```python
class SpikingDecoder(nn.Module):
    def __init__(self, config):
        self.config = config
        
        # Use arctangent surrogate gradient
        spike_grad = surrogate.atan(alpha=2.0)
        
        # LIF processing layer
        self.fc_decode = nn.Linear(hidden_dim, hidden_dim)
        self.lif_decode = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=spike_grad,
            learn_beta=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
    
    def forward(self, encoded_state):
        mem = self.lif_decode.init_leaky()
        decode_history = []
        
        # Temporal decoding (shorter than encoding)
        for t in range(num_time_steps // 2):
            cur = self.fc_decode(encoded_state)
            spk, mem = self.lif_decode(cur, mem)
            decode_history.append(spk)
        
        # Stack and average spike history
        decode_spikes = torch.stack(decode_history, dim=0)
        final_state = decode_spikes.mean(dim=0)
        
        # Project to vocabulary
        logits = self.output_proj(final_state)
        
        return logits, decode_spikes
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_dim | 512 | Hidden dimension |
| vocab_size | 50257 | Output vocabulary size |
| num_time_steps | 10 | Decoding steps (half of encoding) |
| beta | 0.9 | Membrane decay |
| spike_grad | atan(α=2.0) | Surrogate gradient |

## Output

- `logits`: [batch, vocab_size] - Token prediction scores
- `decoder_spikes`: [timesteps, batch, hidden_dim] - Spike history

## Design Choices

1. **Shorter decoding**: Half the time steps of encoding
2. **Arctangent surrogate**: Smooth gradients for stable training
3. **Learnable beta**: Allows adaptation of decay dynamics
4. **GELU activation**: Non-linearity in output projection

## Related Entities

- [[wiki/entities/brain-coding-model|BrainCodingModel]]
- [[wiki/entities/spiking-encoder|SpikingEncoder]]
- [[wiki/entities/associative-memory-module|AssociativeMemory]]

## Related Concepts

- [[wiki/concepts/leaky-integrate-and-fire|Leaky Integrate-and-Fire]]
- [[wiki/concepts/surrogate-gradient|Surrogate Gradient]]
- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
