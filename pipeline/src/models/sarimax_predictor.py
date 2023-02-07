import numpy as np
import pandas as pd
import statsmodels.api as sm

HYPER = {
    'aic': (4, 1, 2, 0, 1, 1, 14),
    'mse': (4, 1, 2, 1, 1, 1, 7)
}


def sarimax_forecast(df: pd.DataFrame, lag: int, error: str) -> np.array:
    p, d, q, P, D, Q, s = HYPER[error]
    model = sm.tsa.statespace.SARIMAX(
        df['target'], order=(p, d, q), seasonal_order=(P, D, Q, s), trend='ct'
    ).fit(disp=-1, maxiter=70)
    forecast = model.forecast(lag).values
    return forecast
