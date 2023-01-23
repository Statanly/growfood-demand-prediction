from copy import deepcopy
from typing import Tuple, Dict, Any

from loguru import logger
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from models import LstmMlp


class PredictiveModel:
    def __init__(self, n_params: int):
        self._n_params = n_params
        self._model = LstmMlp(n_params)

    def __maybe_upd_best_model(self, metric: float, model: nn.Module):
        if metric < self._best_smape:
            self._best_smape = metric
            self._best_model = deepcopy(model.state_dict())

    def __reset_best_model(self):
        self._best_smape = float('inf')
        self._best_model = self._model

    def __df_to_tensor(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # drop last day y
        print(df.head())
        y = df.target.iloc[:-1].to_numpy()

        df = df.drop(['date', 'target'], axis=1)

        # drop last day x
        x_test = torch.Tensor(df.to_numpy()).unsqueeze(0)

        x_train = df.iloc[:-1, :].to_numpy()
        x_train = torch.Tensor(x_train).unsqueeze(0)
        y_train = torch.Tensor(y)

        print(y_train)
        print(x_train.size(), x_test.size(), y_train.size())

        return x_train, y_train, x_test

    def fit_predict(self, df: pd.DataFrame) -> float:
        loss = nn.L1Loss()
        torch.manual_seed(42)
        self._model = LstmMlp(self._n_params)
        opt = optim.Adadelta(self._model.parameters())

        x_train, y_train, x_test = self.__df_to_tensor(df)

        self._model.train()
        self.__reset_best_model()
        for epoch in tqdm(range(500)):
            opt.zero_grad()

            pred = self._model(x_train)
            err = loss(y_train, pred).mean()
            err.backward()
            opt.step()

            err_smape = ((pred - y_train).abs() / (pred.abs() + y_train.abs()) / 2).mean().item()

            self.__maybe_upd_best_model(metric=err_smape, model=self._model)

        logger.info(f'best smape: {self._best_smape:.3f}')
        self._model.load_state_dict(self._best_model)
        self._model.eval()

        pred = self._model.forward(x_test)
        pred = float(pred[0, -1].item())
        return pred

    # def predict(self, df: pd.DataFrame, n_days: int = 1) -> Dict[str, Any]:
    #     x, y = self.__df_to_tensor(df)
    #     with torch.no_grad():
    #         pred = self._model(x)
    #     loguru.info(f'pred {pred.numpy()}')
    #     return {'pred': pred}


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../df.csv')
    model = PredictiveModel(n_params=14)
    res = model.fit_predict(df)
    print('res', res)
