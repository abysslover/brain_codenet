---
title: System Architecture
created: 2026-04-15
updated: 2026-04-15
tags: [synthesis, architecture, overview]
sources: []
---

# System Architecture

Complete architecture overview of the brain emulation-based coding Q&A system.

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Brain Coding Q&A System                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │  Input Text  │───▶│   Spiking    │───▶│   Associative│         │
│  │  (Question)  │    │   Encoder    │    │   Memory     │         │
│  └──────────────┘    │  (Sensory)   │    │  (Hippocampus)│        │
│                      └──────────────┘    └──────────────┘         │
│                            │                   │                    │
│                            ▼                   ▼                    │
│                      ┌──────────────────────────────┐              │
│                      │      Neural Representation   │              │
│                      │     (Sparse Spike Patterns)  │              │
│                      └──────────────────────────────┘              │
│                                    │                               │
│                                    ▼                               │
│                      ┌──────────────┐    ┌──────────────┐         │
│                      │   Sparse     │───▶│   Output     │         │
│                      │   Decoder    │    │   Tokens     │         │
│                      │  (Motor)     │    │  (Answer)    │         │
│                      └──────────────┘    └──────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Spiking Encoder (Sensory Cortex Emulation)

**Purpose**: Convert text to temporal spike patterns

**Flow**:
```
Tokens → Embedding → Rate Coding → LIF Layer 1 → LIF Layer 2 → Encoded
```

**Key Features**:
- Rate coding for stimulus intensity encoding
- Two-layer LIF processing for feature extraction
- Temporal integration over 20 timesteps
- Surrogate gradients for backpropagation

**Parameters**: ~3M

### 2. Associative Memory (Hippocampus Emulation)

**Purpose**: Content-addressable memory retrieval

**Flow**:
```
Query → Similarity → Top-k → Weighted Retrieval → Retrieved
```

**Key Features**:
- Modern Hopfield Network implementation
- Sparse top-k memory selection (k=8)
- Colocated memory keys and values
- Residual connections for information preservation

**Parameters**: ~1M

### 3. Spiking Decoder (Motor Cortex Emulation)

**Purpose**: Convert neural state to output tokens

**Flow**:
```
State → LIF Processing → Spike History → Average → Vocabulary
```

**Key Features**:
- Shorter temporal processing (10 timesteps)
- Arctangent surrogate gradient
- Learnable membrane decay
- GELU-activated output projection

**Parameters**: ~1M

## Information Flow

1. **Encoding Phase**: Text tokens are converted to embedding vectors, then to spike trains via rate coding. Two LIF layers process these temporally, integrating information across 20 timesteps.

2. **Memory Retrieval Phase**: The encoded state is used to query associative memory. Cosine similarity identifies the top-k most relevant stored patterns, which are combined via attention-weighted sum.

3. **Decoding Phase**: The retrieved memory state undergoes temporal processing through a LIF layer, producing spike history that is averaged and projected to vocabulary logits.

## Energy Efficiency Mechanisms

| Mechanism | Implementation | Benefit |
|-----------|---------------|---------|
| Colocation | Memory keys/values in same layer as computation | Eliminates fetch costs |
| Sparse Spiking | LIF neurons fire only above threshold | Reduces active neurons |
| Event-Driven | Computation on spike events only | No idle computation |
| Temporal Integration | Information across time steps | Efficient representation |

## Training Objective

```
Total Loss = Task Loss + λ × Energy Regularization

Task Loss = CrossEntropy(logits, labels)
Energy Loss = (firing_rate - target)^2 + (decoder_energy - target)^2
```

## Scalability

- **Parameters**: ~5M total (vs. billions in large Transformers)
- **Memory**: O(N×D) for memory module (N=memory_size, D=hidden_dim)
- **Compute**: O(T×N×D) per timestep (T=time_steps)

## Related Pages

- [[wiki/concepts/colocation|Colocation]]
- [[wiki/concepts/sparse-spiking|Sparse Spiking]]
- [[wiki/concepts/associative-memory|Associative Memory]]
- [[wiki/entities/brain-coding-model|BrainCodingModel]]
