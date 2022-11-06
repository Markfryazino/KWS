import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Optional

from src.configs import TaskConfig, StreamingTaskConfig
from src.util_classes import LogMelspec


class Attention(nn.Module):

    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.Tanh(),
            nn.Linear(bottleneck_size, 1)
        )

    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)

        return (input * alpha).sum(dim=-2), energy

class CRNN(nn.Module):
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
            config.stride[0] + 1
        
        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size, config.bottleneck_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, input, hidden_state: Optional[torch.Tensor] = None, return_hidden: bool =False, return_alpha: bool =False):
        
        if hidden_state is not None and torch.numel(hidden_state) == 0:
            hidden_state = None
        
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, gru_hidden = self.gru(conv_output, hidden_state)
        contex_vector, energy = self.attention(gru_output)
        output = self.classifier(contex_vector)
        if return_hidden:
            return output, gru_hidden

        if return_alpha:
            return output, energy

        return (output, torch.tensor(0))


class StreamingCRNN(nn.Module):
    def __init__(self, model: CRNN, max_window_length, streaming_step_size, share_hidden_states, device):
        super().__init__()
        self.model = model
        self.max_window_length = max_window_length
        self.streaming_step_size = streaming_step_size
        self.share_hidden_states = share_hidden_states
        self.device = device
        self.melspec = LogMelspec(is_train=False, config=StreamingTaskConfig)

        self.hidden_state = torch.FloatTensor([])
        self.frames = torch.FloatTensor([])
        self.prediction = torch.tensor([1, 0])

    def forward(self, input: torch.FloatTensor):
        self.frames = torch.cat([self.frames, input.flatten()], dim=0)
        if len(self.frames) >= self.max_window_length:

            batch = torch.tensor(self.frames).unsqueeze(dim=0).to(self.device)
            batch = self.melspec(batch)
            logits, out_hidden_state = self.model(batch, self.hidden_state, return_hidden=True)

            if self.share_hidden_states:
                self.hidden_state = out_hidden_state

            self.prediction = F.softmax(logits, dim=-1)[0]

            self.frames = self.frames[-(self.max_window_length - self.streaming_step_size):]

        return self.prediction.cpu()
