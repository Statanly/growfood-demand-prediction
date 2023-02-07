from typing import Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class TsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int):
        not_x_cols = ['date', 'target']
        self._records_x = list(df.drop(not_x_cols, axis=1).to_records(index=False))
        self._records_y = df.target.tolist()
        self._wsize = window_size

    def __len__(self):
        return len(self._records_y) - self._wsize

    def __getitem__(self, ix: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._records_x[ix: ix + self._wsize]
        y = self._records_y[ix: ix + self._wsize]

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        return x, y

    @staticmethod
    def collate_fn(samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*samples)
        xs = torch.stack(xs)
        ys = torch.stack(ys)

        return xs, ys
