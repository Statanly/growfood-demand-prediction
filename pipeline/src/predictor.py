from typing import Dict, Any

import pandas as pd

from pipeline.models.lstm_predictor import LstmPredictor
from pipeline.transforms import (
    MovingAvgTransform,
    WeekdayTransform, HoltWintersPredictTransform, IsWeekendTransform
)


class Predictor:
    _transforms = [
        HoltWintersPredictTransform(name='hws_1d', column='target'),
        WeekdayTransform(name='weekday', column='date'),
        IsWeekendTransform(name='weekend', column='date'),
        MovingAvgTransform(name='ma_7', column='target', value=7),
        MovingAvgTransform(name='ma_14', column='target', value=14),
    ]

    def __init__(self):
        self._model = LstmPredictor(n_params=9)

    def __transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for tr in self._transforms:
            df = tr.transform(df)
        return df

    def fit_predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self.__transform(df)
        # df.to_csv('./df.csv', index=False)
        res = self._model.fit_predict(df.iloc[7:, :])
        # res = self._model.predict(df, n_days=1)
        return {'orders_count': int(res)}
