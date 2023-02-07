import torch
from torch import nn


class LstmMlp(nn.Module):
    def __init__(self, features_n: int):
        super().__init__()
        self._rec = nn.LSTM(input_size=features_n, hidden_size=128, num_layers=1, batch_first=True)
        self._lin = nn.Sequential(
            self.__lin_block(128, 64),
            nn.Linear(64, 1),
            nn.LeakyReLU(0.1)
        )

    def __lin_block(self, size_in: int, size_out: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self._rec(x)

        # reshape for linear layer
        batch, length, features = out.size()
        out = out.reshape(batch * length, features)
        out = self._lin(out)

        # reshape back to sequence
        out = out.view(batch, length)
        return out
