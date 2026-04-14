import torch
import torch.nn.functional as F


def rate_encoding(embeddings: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Convert embeddings to spike trains via rate coding.
    Emulates how neurons encode stimulus intensity as firing rate.

    Args:
        embeddings: [batch, seq_len, embed_dim]
        num_steps: Number of time steps
    Returns:
        spike_trains: [num_steps, batch, seq_len, embed_dim]
    """
    prob = torch.sigmoid(embeddings)

    spike_trains = []
    for _ in range(num_steps):
        spikes = torch.bernoulli(prob)
        spike_trains.append(spikes)

    return torch.stack(spike_trains, dim=0)


def compute_energy_metrics(spike_train: torch.Tensor, fan_out: int = None) -> dict:
    """
    Compute metrics simulating brain energy efficiency.

    Args:
        spike_train: Spike train tensor [T, B, H] or [B, L, H]
        fan_out: Output dimension of next layer (for SOP calculation)

    Returns:
        dict: Contains sparsity, firing_rate, total_spikes, energy_reduction,
              snn_sops, dense_macs, energy_ratio_physical
    """
    total_possible = spike_train.numel()
    active_spikes = spike_train.sum().item()

    # Basic sparsity metrics
    firing_rate = active_spikes / total_possible
    sparsity = 1.0 - firing_rate

    # Theoretical energy reduction based on event-driven processing: E ∝ firing_rate
    energy_reduction = 1.0 / max(firing_rate, 1e-6)

    metrics = {
        "sparsity": sparsity,
        "firing_rate": firing_rate,
        "total_spikes": active_spikes,
        "energy_reduction": energy_reduction,
    }

    # SOP vs MAC calculation with physical grounding
    if fan_out is not None:
        # Dense model MAC operations (all neurons compute)
        dense_macs = total_possible * fan_out

        # SNN SOP operations (only spikes compute)
        snn_sops = active_spikes * fan_out

        # Physical energy ratio (45nm CMOS reference)
        # MAC: ~4.6pJ, ADD: ~0.9pJ
        if dense_macs > 0:
            energy_ratio_physical = (snn_sops * 0.9) / (dense_macs * 4.6)
        else:
            energy_ratio_physical = 1.0

        metrics.update(
            {
                "snn_sops": snn_sops,
                "dense_macs": dense_macs,
                "energy_ratio_physical": energy_ratio_physical,
            }
        )

    return metrics
