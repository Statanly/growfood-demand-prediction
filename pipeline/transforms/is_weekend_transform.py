from datetime import timedelta

import pandas as pd

from .transform import Transform


class IsWeekendTransform(Transform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_column_name = self.get_name()

        assert self._column in df.columns
        assert out_column_name not in df.columns

        weekends = df[self._column].apply(lambda d: (d + timedelta(days=3)).weekday() < 5)
        weekends = pd.get_dummies(weekends, prefix=out_column_name)
        df = pd.concat((df, weekends), axis=1)
        return df
