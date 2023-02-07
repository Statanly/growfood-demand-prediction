import pandas as pd
import torch
import random

import numpy as np

from etna.datasets.tsdataset import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform, PytorchForecastingTransform
from etna.transforms import LagTransform
from etna.models.nn import DeepARModel

from pytorch_forecasting.data import GroupNormalizer

import warnings
warnings.filterwarnings("ignore")


def etna_forecast(df: pd.DataFrame, lag: int) -> np.array:
    df = df.drop(['new_orders_count', 'custom_orders_rate'], axis=1)
    df = df.rename({'date': 'timestamp'}, axis=1).assign(segment='main')
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    transform_date = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="dateflag")
    num_lags = 14
    transform_lag = LagTransform(
        in_column="target",
        lags=[lag + i for i in range(num_lags)],
        out_column="target_lag",
    )
    lag_columns = [f"target_lag_{lag + i}" for i in range(num_lags)]

    transform_deepar = PytorchForecastingTransform(
        max_encoder_length=lag,
        max_prediction_length=lag,
        time_varying_known_reals=["time_idx"] + lag_columns,
        time_varying_unknown_reals=["target"],
        time_varying_known_categoricals=["dateflag_day_number_in_week"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    model_deepar = DeepARModel(max_epochs=150, learning_rate=[0.01], gpus=0, batch_size=64)
    pipeline_deepar = Pipeline(
        model=model_deepar,
        horizon=lag,
        transforms=[transform_lag, transform_date, transform_deepar],
    )

    model = pipeline_deepar.fit(ts)
    forecast = model.forecast(n_folds=1)
    return forecast.df.values[:, 2]
