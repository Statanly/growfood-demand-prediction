from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy import sqrt
from torch import optim, nn
from tqdm import tqdm

from pipeline.transforms import *


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


class LstmPredictor:
    def __init__(self, n_params: int):
        self._n_params = n_params
        self._model = LstmMlp(n_params)
        self._opt = optim.Adadelta(self._model.parameters(), lr=sqrt(5))
        self._sched = optim.lr_scheduler.StepLR(self._opt, 50, 0.5)
        self._best_smape = float('inf')
        self._best_model = None

    def __maybe_upd_best_model(self, metric: float):
        if metric < self._best_smape:
            self._best_smape = metric
            self._best_model = deepcopy(self._model.state_dict())

    def __transform(self, df, lag):
        for transformer in [
            HoltWintersPredictTransform(name='hws_1d', column='target'),
            WeekdayTransform(name='weekday', column='date'),
            IsWeekendTransform(name='weekend', column='date'),
            MovingAvgTransform(name='ma_7', column='target', value=7),
            MovingAvgTransform(name='ma_14', column='target', value=14),
        ]:
            df = transformer.transform(df)
        y = df.target[lag:].to_numpy()
        df = df.drop(['date', 'target'], axis=1)
        x = df.iloc[:-lag, :].to_numpy()
        return df, x, y

    def fit_predict(self, df: pd.DataFrame, lag: int):
        df, x, y = self.__transform(df, lag)
        x, y = torch.Tensor(x).unsqueeze(0), torch.Tensor(y)

        loss = nn.L1Loss(reduction='mean')
        self._model.train()
        for _ in tqdm(range(400)):
            self._opt.zero_grad()
            pred = self._model(x)
            err = loss(y, pred)
            err.backward()
            self._opt.step()
            self._sched.step()
            err_mape = ((pred - y).abs() / y.abs()).mean()
            self.__maybe_upd_best_model(metric=err_mape.item())
        logger.info(f'best smape: {self._best_smape:.3f}')

        self._model.load_state_dict(self._best_model)
        self._model.eval()

        fore_x = df.iloc[lag:, :].to_numpy()
        fore_x = torch.Tensor(fore_x).unsqueeze(0)
        forecast = self._model(fore_x)[0]
        return forecast.detach().numpy()[-lag:]
