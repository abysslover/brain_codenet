# Brain Emulation Coding Q&A System

A PyTorch-based implementation of a brain-inspired neural architecture for coding Q&A tasks. This system emulates biological brain principles to achieve energy-efficient computation through sparse spiking, colocation, and event-driven processing.

## Motivation

Current AI hardware faces a fundamental energy efficiency problem:

- **Human brain**: ~20W power, ~1 exaFLOP operations
- **Frontier supercomputer**: 20MW for same performance (1 million× more energy)

This gap stems from the **Von Neumann bottleneck** - the separation of computation and memory creates a "Memory Wall" where data movement costs 10,000× more than actual computation.

## Core Principles

### 1. Colocation
Memory storage and computation occur in the same physical space, eliminating fetch-compute-discard cycles. Unlike traditional Transformers, this system uses **associative memory** for information retrieval.

### 2. Sparse Spiking Communication
Instead of dense activation across all neurons at every timestep, only neurons exceeding a threshold fire spikes asynchronously, consuming power only when active.

### 3. Event-Driven Processing
Computation triggers only on spike events rather than continuous matrix operations, simulating the brain's energy-efficient processing.

## Architecture

```
[Spiking Encoder] → [SNN Core] → [Associative Memory] → [Sparse Decoder]
       ↑                ↓              ↓                    ↓
   Question Input   Cortical Column  Hippocampal      Response Generation
                    Processing     Associative Memory
```

### Module Descriptions

#### 1. Spiking Encoder (Sensory Cortex Emulation)
- Converts text tokens to embedding vectors
- Transforms embeddings into temporal spike trains via rate coding
- Uses Leaky Integrate-and-Fire (LIF) neurons with surrogate gradients

#### 2. SNN Core (Cortical Column Emulation)
- Implements LIF neuron dynamics with discrete time steps
- Membrane potential evolution: `V[t+1] = β·V[t] + W·S_in[t]`
- Spike output: `S_out[t] = Θ(V[t+1] - V_th)`
- Surrogate gradients enable backpropagation through spikes

#### 3. Associative Memory (Hippocampus Emulation)
- Modern Hopfield Network for content-addressable memory
- Sparse top-k memory retrieval mimicking biological attention
- Colocated memory keys and values enable unified computation

#### 4. Sparse Decoder (Motor Cortex Emulation)
- Integrates SNN and Memory outputs
- Maintains sparse activation patterns
- Generates code token predictions

## Energy Efficiency Metrics

Since actual power consumption is hard to measure, we use proxy metrics:

- **Sparsity**: `1 - (active_spikes / total_possible)`
- **Firing Rate**: `active_spikes / total_possible`
- **Energy Loss**: Weighted combination of spike count, memory accesses, and activation L1 norm

## Project Structure

```
brain_coding_qa/
├── src/
│   ├── config.py              # Configuration settings
│   ├── data/
│   │   ├── dataset.py         # HuggingFace dataset loader
│   │   └── spike_encoding.py  # Rate coding utilities
│   ├── models/
│   │   ├── spiking_encoder.py # LIF-based encoder
│   │   ├── memory_module.py   # Associative memory
│   │   ├── snn_decoder.py     # Spiking decoder
│   │   └── brain_model.py     # Integrated model
│   ├── training/
│   │   └── trainer.py         # Training pipeline
│   └── main.py                # Entry point
├── wiki/                      # Documentation
└── requirements.txt           # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

### Configuration

Edit `config.py` to customize:

```python
# Dataset
dataset_name: str = "iamtarun/python_code_instructions_18k_alpaca"
max_samples: int = 1000
max_input_length: int = 128
max_output_length: int = 64

# SNN Parameters
vocab_size: int = 50257
embed_dim: int = 256
snn_hidden_dim: int = 512
num_time_steps: int = 20
beta: float = 0.9
threshold: float = 1.0

# Training
batch_size: int = 8
learning_rate: float = 3e-4
num_epochs: int = 15
energy_loss_weight: float = 0.01
```

## Training Output

```
============================================================
  Brain Emulation Coding Q&A System
============================================================
[Data] Number of batches: 125
[Model] Total parameters: 12,458,304
[Trainer] Device: cuda
[Trainer] Parameters: 12,458,304

=== Starting Brain Emulation Training ===
Epoch 1: loss=2.3456, energy=0.1234, sparsity=0.892
Epoch 2: loss=1.9876, energy=0.0987, sparsity=0.912
...

=== Training Complete ===

[Inference Test]
Question: Write a Python function that calculates the factorial of a number.
Predicted First Token: 'def'
Spike Sparsity: 0.9234
Total Spikes: 15680
```

## Implementation Details

### Leaky Integrate-and-Fire Neuron

```python
class SpikingEncoder(nn.Module):
    def __init__(self, config):
        self.fc1 = nn.Linear(config.embed_dim, config.snn_hidden_dim)
        self.lif1 = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=surrogate.fast_sigmoid(slope=config.spike_grad_slope),
            learn_beta=True
        )
    
    def forward(self, input_ids, attention_mask):
        # Process through temporal steps
        for t in range(self.config.num_time_steps):
            cur = self.fc1(x_t)
            spk, mem = self.lif1(cur, mem)
```

### Associative Memory

```python
class AssociativeMemory(nn.Module):
    def forward(self, query):
        # Cosine similarity search
        similarity = torch.matmul(query_norm, keys_norm.t())
        
        # Sparse top-k retrieval
        top_scores, top_indices = torch.topk(similarity, self.top_k, dim=-1)
        
        # Weighted memory combination
        retrieved = sum(weights * selected_memories)
```

## Future Extensions

1. **Autoregressive Decoding**: Extend from single-token to sequential generation
2. **Temporal Attention**: Strengthen temporal dependencies in spike sequences
3. **Neuromorphic Hardware**: Port to Intel Loihi or SpiNNaker chips
4. **Code Execution Feedback**: Incorporate runtime results into training

## References

- Pillow, T. et al. "Energy-Efficient Computing with Spiking Neural Networks"
- Kheradpisheh, S.R. et al. "STDP-Based Spiking Deep Learning"
- Merolla, P. et al. "A Million Spiking-Neuron Integrated Circuit"

## License

MIT License - See LICENSE file for details.
