import pandas as pd

from .transform import Transform


class WeekdayTransform(Transform):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_column_name = self.get_name()

        assert self._column in df.columns
        assert out_column_name not in df.columns

        weekdays = df[self._column].apply(lambda d: d.weekday())
        weekdays = pd.get_dummies(weekdays, prefix=out_column_name)
        df = pd.concat((df, weekdays), axis=1)
        return df
