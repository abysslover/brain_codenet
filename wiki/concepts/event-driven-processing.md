---
title: Event-Driven Processing
created: 2026-04-15
updated: 2026-04-15
tags: [principle, computation, efficiency]
sources: []
---

# Event-Driven Processing

Event-driven processing executes computation only when triggered by events (spikes), rather than continuous operations.

## Overview

### Traditional Deep Learning (Continuous)

```
For each timestep:
    Perform full matrix multiplication
    Update all activations
    # Even if inputs are zero, computation happens
```

### Event-Driven (Brain-Inspired)

```
Wait for spike events:
    If spike received:
        Update connected neurons
        Perform computation
    Else:
        Decay membrane potential only
        # Minimal energy consumption
```

## Biological Basis

In biological brains:

- Neurons are mostly idle between spikes
- Synaptic transmission occurs only on spike events
- Resting membrane potential decay is energy-efficient
- Active computation happens only when needed

## Implementation

```python
class SpikingDecoder(nn.Module):
    mem = self.lif_decode.init_leaky()
    decode_history = []
    
    # Computation only when spikes occur
    for t in range(num_time_steps // 2):
        cur = self.fc_decode(encoded_state)
        spk, mem = self.lif_decode(cur, mem)  # Only compute on spike
        decode_history.append(spk)
```

## Advantages

1. **Energy Efficiency**: No computation during idle periods
2. **Scalability**: Computation scales with activity, not network size
3. **Latency**: Responses triggered immediately by events
4. **Asynchrony**: No global clock needed

## Applications

- Sensory processing (vision, audition)
- Event-based cameras
- Sparse attention mechanisms
- Neuromorphic hardware (Loihi, SpiNNaker)

## Related Concepts

- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/concepts/colocation|Colocation]]
- [[wiki/concepts/energy-efficiency|Energy Efficiency]]
- [[wiki/entities/spiking-decoder|SpikingDecoder]]
