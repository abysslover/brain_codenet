import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from data.spike_encoding import rate_encoding, compute_energy_metrics


class SpikingEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        spike_grad = surrogate.fast_sigmoid(slope=config.spike_grad_slope)

        self.fc1 = nn.Linear(config.embed_dim, config.snn_hidden_dim)
        self.lif1 = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        self.fc2 = nn.Linear(config.snn_hidden_dim, config.snn_hidden_dim)
        self.lif2 = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            encoded_state: [batch_size, snn_hidden_dim]
            energy_stats: dict
        """
        batch_size, seq_len = input_ids.shape

        x = self.embedding(input_ids)
        x = self.dropout(x)

        spike_trains = rate_encoding(x, self.config.num_time_steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1_history = []
        spk2_history = []

        for t in range(self.config.num_time_steps):
            x_t = spike_trains[t]

            mask = attention_mask.unsqueeze(-1).float()
            x_t = x_t * mask

            x_t = x_t.mean(dim=1)

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_history.append(spk1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_history.append(spk2)

        spk2_stack = torch.stack(spk2_history, dim=0)
        encoded_state = spk2_stack.mean(dim=0)

        energy_stats = compute_energy_metrics(
            spk2_stack, fan_out=self.config.snn_hidden_dim
        )
        energy_stats["spike_stack"] = spk2_stack

        return encoded_state, energy_stats
