import dataclasses
import torch

from typing import Tuple


@dataclasses.dataclass
class TaskConfig:
    melspec_n_fft: int = 400
    melspec_win_length: int = 400
    melspec_hop_length: int = 160
    random_state: int = 42
    keyword: str = 'sheila'
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 20
    n_mels: int = 40
    cnn_out_channels: int = 8
    kernel_size: Tuple[int, int] = (5, 20)
    stride: Tuple[int, int] = (2, 8)
    hidden_size: int = 32
    bottleneck_size: int = 32
    gru_num_layers: int = 2
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class DistillTaskConfig(TaskConfig):
    distill_w: float = 1.0
    attn_distill_w: float = 0.0


@dataclasses.dataclass
class StreamingTaskConfig(TaskConfig):
    # max_window_length и streaming_step_size у нас будут в секундах
    max_window_length_seconds: float = 1.
    streaming_step_size_seconds: float = 0.1
    share_hidden_states: bool = True

    def __post_init__(self):
        self.max_window_length = int(self.max_window_length_seconds * self.sample_rate)
        self.streaming_step_size = int(self.streaming_step_size_seconds * self.sample_rate)
