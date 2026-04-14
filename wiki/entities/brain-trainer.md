---
title: BrainTrainer
created: 2026-04-15
updated: 2026-04-15
tags: [entity, trainer, training]
sources: []
---

# BrainTrainer

The BrainTrainer manages the training pipeline for the brain emulation model, including optimization and monitoring.

## Role

Orchestrates the training loop, computes metrics, and tracks energy efficiency throughout training.

## Implementation

```python
class BrainTrainer:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_energy = 0
        total_sparsity = 0
        
        for batch in dataloader:
            # Forward pass
            output = model(input_ids, attention_mask, labels)
            loss = output["loss"]
            
            # Backward pass
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            total_energy += output["energy_loss"].item()
            total_sparsity += output["encoder_stats"]["sparsity"]
        
        scheduler.step()
        
        return {
            "avg_loss": total_loss / num_batches,
            "avg_energy": total_energy / num_batches,
            "avg_sparsity": total_sparsity / num_batches
        }
    
    def train(self):
        for epoch in range(num_epochs):
            stats = self.train_epoch(epoch)
            # Log progress
```

## Training Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| loss | Task loss (cross-entropy) | Decreasing |
| energy | Energy regularization | Low (~0.05-0.1) |
| sparsity | Spike sparsity | High (>0.9) |
| lr | Learning rate | Cosine decay |

## Optimization Strategy

1. **AdamW**: Weight decay regularization
2. **Gradient Clipping**: Prevent exploding gradients (norm ≤ 1.0)
3. **Warm Restarts**: Cosine annealing with periodic restarts
4. **Energy Regularization**: Balance task performance with efficiency

## Output Format

```
=== Starting Brain Emulation Training ===
Epoch 1: loss=2.3456, energy=0.1234, sparsity=0.892
Epoch 2: loss=1.9876, energy=0.0987, sparsity=0.912
...
=== Training Complete ===
```

## Related Entities

- [[wiki/entities/brain-coding-model|BrainCodingModel]]

## Related Concepts

- [[wiki/concepts/energy-efficiency|Energy Efficiency Metrics]]
- [[wiki/concepts/surrogate-gradient|Surrogate Gradient]]
