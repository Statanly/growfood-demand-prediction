import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm

from .transform import Transform


class HoltWintersPredictTransform(Transform):
    def __preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ['date', self._column]
        line = df[cols]#.copy()

        line.index = pd.to_datetime(line['date'])
        line = line.drop('date', axis=1)
        line = line.asfreq('1d')

        return line

    @staticmethod
    def __fit_predict_hws_1d(line: pd.DataFrame) -> float:
        best_params = {
            'use_boxcox': False,
            'trend': 'add',
            'damped_trend': False,
            'seasonal': 'add',
            'seasonal_periods': 7,
            'freq': 'D'
        }

        fit1 = ExponentialSmoothing(line, **best_params)
        fit1 = fit1.fit()

        pred = fit1.forecast(1).item()
        return pred

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_column_name = self.get_name()
        # transform df forth
        line = self.__preprocess_data(df)

        assert self._column in df.columns
        assert out_column_name not in df.columns

        hisory_size = 7 * 7 + 1
        # min forcast history size. Less make model inaccurate
        default_value = line[self._column].iloc[0].item()
        padding = [default_value for _ in range(abs(hisory_size))]
        vals = line[self._column]

        # i-1 to avoid data leak
        recenlty = lambda i: max(hisory_size, i) - min(hisory_size, i)

        pred = []
        for i in tqdm(range(hisory_size, len(vals))):
            pred.append(self.__fit_predict_hws_1d(vals.iloc[recenlty(i): i - 1]))
        pred = padding + pred

        df.loc[:, out_column_name] = pred
        print('cols', df.columns)
        # transform df back
        # df = df.reset_index()
        return df
