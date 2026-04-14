import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spiking_encoder import SpikingEncoder
from models.memory_module import AssociativeMemory
from models.snn_decoder import SpikingDecoder


class BrainCodingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = SpikingEncoder(config)
        self.memory = AssociativeMemory(config)
        self.decoder = SpikingDecoder(config)

        self.energy_regularizer = EnergyRegularizer(config)

    def forward(self, input_ids, attention_mask, labels=None):
        encoded_state, encoder_stats = self.encoder(input_ids, attention_mask)

        # Ablation: Memory 모듈 on/off
        if self.config.use_associative_memory:
            memory_state, attention_weights = self.memory(encoded_state)
        else:
            memory_state = encoded_state
            attention_weights = torch.zeros(
                encoded_state.size(0), 1, device=encoded_state.device
            )

        logits, decoder_spikes = self.decoder(memory_state)

        output = {
            "logits": logits,
            "encoder_stats": encoder_stats,
            "decoder_spikes": decoder_spikes,
            "attention_weights": attention_weights,
        }

        if labels is not None:
            task_loss = F.cross_entropy(
                logits, labels[:, 0], ignore_index=self.config.pad_token_id
            )

            # Ablation: Energy regularizer on/off
            if self.config.use_energy_regularizer:
                energy_loss = self.energy_regularizer(encoder_stats, decoder_spikes)
                total_loss = task_loss + self.config.energy_loss_weight * energy_loss
            else:
                energy_loss = torch.tensor(0.0, device=logits.device)
                total_loss = task_loss

            output.update(
                {"loss": total_loss, "task_loss": task_loss, "energy_loss": energy_loss}
            )

        return output


class EnergyRegularizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sparsity_target = config.sparsity_target

    def forward(self, encoder_stats, decoder_spikes):
        encoder_sparsity_loss = (
            encoder_stats["firing_rate"] - self.sparsity_target
        ) ** 2

        decoder_energy = decoder_spikes.float().mean()
        decoder_sparsity_loss = (decoder_energy - self.sparsity_target) ** 2

        return encoder_sparsity_loss + decoder_sparsity_loss
