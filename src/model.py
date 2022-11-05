import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs import TaskConfig, StreamingTaskConfig
from src.util_classes import LogMelspec


class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)

        return (input * alpha).sum(dim=-2), alpha

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

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, input, hidden_state=None, return_hidden=False, return_alpha=False):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, gru_hidden = self.gru(conv_output, hidden_state)
        contex_vector, alpha = self.attention(gru_output, return_alpha)
        output = self.classifier(contex_vector)
        if return_hidden:
            return output, gru_hidden

        if return_alpha:
            return output, alpha

        return output


class StreamingCRNN(nn.Module):
    def __init__(self, model: CRNN, config: StreamingTaskConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.melspec = LogMelspec(is_train=False, config=StreamingTaskConfig)

        self.hidden_state = None
        self.frames = []
        self.prediction = torch.tensor([1, 0])

    def reset(self):
        self.hidden_state = None
        self.frames = []
        self.prediction = torch.tensor([1, 0])

    def forward(self, input):
        self.frames.append(input)
        if len(self.frames) == self.config.max_window_length:

            batch = torch.tensor(self.frames).unsqueeze(dim=0).to(self.config.device)
            batch = self.melspec(batch)
            logits, out_hidden_state = self.model(batch, self.hidden_state, return_hidden=True)

            if self.config.share_hidden_states:
                self.hidden_state = out_hidden_state

            self.prediction = F.softmax(logits, dim=-1)[0]

            self.frames = self.frames[-(self.config.max_window_length - self.config.streaming_step_size):]

        return self.prediction.cpu()
