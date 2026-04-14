---
title: Leaky Integrate-and-Fire
created: 2026-04-15
updated: 2026-04-15
tags: [neuron, model, dynamics]
sources: []
---

# Leaky Integrate-and-Fire

The Leaky Integrate-and-Fire (LIF) neuron is the fundamental computational unit in spiking neural networks, modeling biological neuron behavior.

## Mathematical Model

### Membrane Potential Dynamics

```
dV/dt = -β·V + I_in
```

Discrete time version:

```
V[t+1] = β·V[t] + W·S_in[t]
```

Where:
- `V`: membrane potential
- `β`: decay factor (0 < β ≤ 1)
- `W`: synaptic weights
- `S_in`: input spikes

### Firing Condition

```
if V[t] ≥ V_th:
    S_out = 1  # Spike emitted
    V[t] = V_reset  # Potential reset
else:
    S_out = 0  # No spike
```

## Parameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| β (decay) | 0.8 - 0.95 | Higher = slower decay, more temporal integration |
| V_th (threshold) | 1.0 | Higher = fewer spikes, more selective |
| V_reset | 0.0 | Potential after spike |
| refractory | 1-2 timesteps | Period where neuron cannot fire |

## Implementation

```python
import snntorch as snn

# Create LIF neuron
lif = snn.Leaky(
    beta=0.9,              # 10% decay per timestep
    threshold=1.0,         # Fire at V=1.0
    spike_grad=surrogate.fast_sigmoid(slope=25.0),  # For backprop
    learn_beta=False       # Whether β is trainable
)

# Forward pass
spike, membrane = lif(input_current, membrane)
```

## Role in Brain Emulation

LIF neurons enable:

1. **Temporal Processing**: Memory of past inputs via membrane potential
2. **Sparse Activation**: Only fire when integrated input exceeds threshold
3. **Energy Efficiency**: Low power during idle periods
4. **Biological Plausibility**: Matches real neuron behavior

## Related Concepts

- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/concepts/surrogate-gradient|Surrogate Gradient]]
- [[wiki/entities/spiking-encoder|SpikingEncoder]]
- [[wiki/entities/spiking-decoder|SpikingDecoder]]
