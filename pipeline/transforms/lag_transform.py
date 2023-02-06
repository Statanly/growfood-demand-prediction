import pandas as pd

from .transform import Transform


class LagTransform(Transform):
    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_column_name = self.get_name()
        lag = self._params['value']

        assert self._column in df.columns
        assert out_column_name not in df.columns
        assert lag != 0

        default_value = df[self._column].iloc[0].item()
        padding = [default_value for _ in range(abs(lag))]

        if lag > 0:
            col = padding + df[self._column].iloc[:-lag].tolist()
        else:
            col = df[self._column].iloc[abs(lag):].tolist() + padding

        df.loc[:, out_column_name] = col
        return df
