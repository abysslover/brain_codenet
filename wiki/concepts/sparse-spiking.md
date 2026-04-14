---
title: Sparse Spiking
created: 2026-04-15
updated: 2026-04-15
tags: [principle, activation, efficiency]
sources: []
---

# Sparse Spiking

Sparse spiking is a neural activation strategy where only neurons exceeding a threshold fire, mimicking biological neural efficiency.

## Overview

In conventional neural networks, all neurons activate at every forward pass:

```
Dense activation: [0.1, 0.8, 0.3, 0.9, 0.2]  # All values active
```

Biological brains use **sparse activation**:

```
Sparse spiking:   [0, 1, 0, 1, 0]  # Only threshold-exceeding neurons fire
```

## LIF Neuron Dynamics

Leaky Integrate-and-Fire neurons implement sparse spiking:

```
V[t+1] = β·V[t] + W·S_in[t]  # Membrane potential update
S_out[t] = Θ(V[t+1] - V_th)  # Spike if threshold exceeded
```

Where:
- `V`: membrane potential
- `β`: decay factor (leakage)
- `W`: synaptic weights
- `Θ`: Heaviside step function
- `V_th`: firing threshold

## Implementation

```python
class SpikingEncoder(nn.Module):
    self.lif1 = snn.Leaky(
        beta=config.beta,        # 0.9 - membrane decay
        threshold=config.threshold,  # 1.0 - spike threshold
        spike_grad=surrogate.fast_sigmoid(slope=25.0)
    )
    
    def forward(self, input_ids):
        for t in range(num_time_steps):
            cur = self.fc1(x_t)
            spk, mem = self.lif1(cur, mem)  # Sparse spike output
```

## Energy Implications

- **Dense**: All neurons consume power every timestep
- **Sparse**: Only active neurons consume power
- **Efficiency**: `Energy ∝ (1 - sparsity)`

## Related Concepts

- [[wiki/concepts/colocation|Colocation]]
- [[wiki/concepts/event-driven-processing|Event-Driven Processing]]
- [[wiki/concepts/leaky-integrate-and-fire|Leaky Integrate-and-Fire]]
- [[wiki/concepts/energy-efficiency|Energy Efficiency]]
