import pandas as pd


class Transform:
    def __init__(self, name: str, column: str, **kwargs):
        self._name = name
        self._column = column
        self._params = kwargs

    def get_name(self) -> str:
        params = [str(v) for v in self._params.values()]
        params = '_'.join(sorted(params))
        return f'{self._name}_{params}_{self._column}'

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
