import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        spike_grad = surrogate.atan(alpha=2.0)

        self.fc_decode = nn.Linear(config.snn_hidden_dim, config.snn_hidden_dim)
        self.lif_decode = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=spike_grad,
            learn_beta=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(config.snn_hidden_dim, config.snn_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.snn_hidden_dim // 2, config.vocab_size),
        )

    def forward(self, encoded_state: torch.Tensor):
        mem = self.lif_decode.init_leaky()
        decode_history = []

        for t in range(self.config.num_time_steps // 2):
            cur = self.fc_decode(encoded_state)
            spk, mem = self.lif_decode(cur, mem)
            decode_history.append(spk)

        decode_spikes = torch.stack(decode_history, dim=0)

        final_state = decode_spikes.mean(dim=0)
        logits = self.output_proj(final_state)

        return logits, decode_spikes
