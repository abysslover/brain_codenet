---
title: Surrogate Gradient
created: 2026-04-15
updated: 2026-04-15
tags: [gradient, backpropagation, training]
sources: []
---

# Surrogate Gradient

Surrogate gradients enable backpropagation through non-differentiable spike operations in spiking neural networks.

## The Problem

Spiking neurons use a Heaviside step function:

```
S_out = Θ(V - V_th) = { 0 if V < V_th
                       { 1 if V ≥ V_th
```

The derivative of Θ is zero everywhere except at V = V_th (where it's undefined):

```
dΘ/dV = 0  # Cannot backpropagate!
```

## The Solution: Surrogate Gradients

Replace the true gradient with a differentiable approximation during backprop:

```
True:     dΘ/dV = 0
Surrogate: dΘ̃/dV ≈ Gaussian or sigmoid shape
```

Common surrogate functions:

### Fast Sigmoid

```
σ'(x) = 1 / (1 + |x|)^2
```

### Arctangent

```
σ'(x) = α / (π·(1 + (α·x)^2))
```

### Straight-Through Estimator

```
σ'(x) = 1 if |x| < 1
        0 otherwise
```

## Implementation

```python
from snntorch import surrogate

# Fast sigmoid surrogate
spike_grad = surrogate.fast_sigmoid(slope=25.0)

lif = snn.Leaky(
    beta=0.9,
    threshold=1.0,
    spike_grad=spike_grad  # Use surrogate for backprop
)

# During forward: actual step function
# During backward: surrogate gradient flows
```

## Gradient Flow

```
Forward:  V → Θ(V) → Spike (discontinuous)
Backward: Loss → surrogate'(V) → Gradients (smooth)
```

## Why It Works

1. **Forward pass**: Maintains discrete spike behavior
2. **Backward pass**: Smooth gradients enable learning
3. **Training**: Network learns to produce useful spike patterns
4. **Convergence**: Surrogate shape affects training dynamics

## Related Concepts

- [[wiki/concepts/leaky-integrate-and-fire|Leaky Integrate-and-Fire]]
- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/entities/brain-coding-model|BrainCodingModel]]
