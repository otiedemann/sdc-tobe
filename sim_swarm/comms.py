import random
from dataclasses import dataclass


@dataclass
class CommsModel:
    latency_ms_mean: float
    latency_ms_jitter: float
    packet_loss: float
    dropout_prob_per_tick: float = 0.0

    def command_delivered(self) -> bool:
        return random.random() >= self.packet_loss

    def drone_drops(self) -> bool:
        return random.random() < self.dropout_prob_per_tick
