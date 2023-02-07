from statistics import mean

import pandas as pd

from .transform import Transform


class MovingAvgTransform(Transform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_column_name = self.get_name()
        window = int(self._params['value'])

        assert self._column in df.columns
        assert out_column_name not in df.columns
        assert window > 1
        assert window < len(df) - 1

        default_value = df[self._column].iloc[0].item()
        padding = [default_value for _ in range(abs(window))]
        vals = df[self._column].tolist()

        # i-1 to avoid data leak
        moving_avg = [mean(vals[i - window: i - 1]) for i in range(window, len(vals))]
        moving_avg = padding + moving_avg

        df.loc[:, out_column_name] = moving_avg
        return df
