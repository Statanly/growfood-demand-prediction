import datetime
import warnings

import pandas as pd
from fastapi import APIRouter

from src.models import *
from src.demand_request import DemandRequest

warnings.filterwarnings("ignore")
predict_router = APIRouter()
LAG_TEST = 5


def make_df(features: DemandRequest):
    df = pd.DataFrame.from_dict(dict(features.regressors))

    # lag depends on city
    if features.city_id == 1:
        # spb
        lag = 3
    elif features.city_id == 2:
        # msk
        lag = 4
    else:
        raise ValueError('Unknown city_id')

    date = features.current_date
    dates = [date - datetime.timedelta(days=x) for x in range(len(df), 0, -1)]
    df.loc[:, 'date'] = dates

    df.rename(columns={'deliveries_count': 'target'}, inplace=True)
    cols_order = ['date', 'target', 'new_orders_count', 'custom_orders_rate']
    return df[cols_order], date, lag


@predict_router.post("/predict")
async def predict(features: DemandRequest):
    df, date, lag = make_df(features)
    forecast = LstmPredictor(14).fit_predict(df, lag)
    return {'orders_count': int(forecast[-1])}


@predict_router.post("/predicts/lstm")
async def predict(features: DemandRequest):
    df, date, lag = make_df(features)
    forecast = LstmPredictor(14).fit_predict(df, LAG_TEST)
    return {date + datetime.timedelta(days=x): int(forecast[x]) for x in range(LAG_TEST)}


@predict_router.post("/predicts/etna")
async def predict(features: DemandRequest):
    df, date, lag = make_df(features)
    forecast = etna_forecast(df, LAG_TEST)
    return {date + datetime.timedelta(days=x): int(forecast[x]) for x in range(LAG_TEST)}


@predict_router.post("/predicts/sarimax/aic")
async def predict(features: DemandRequest):
    df, date, lag = make_df(features)
    forecast = sarimax_forecast(df, LAG_TEST, 'aic')
    return {date + datetime.timedelta(days=x): int(forecast[x]) for x in range(LAG_TEST)}


@predict_router.post("/predicts/sarimax/mse")
async def predict(features: DemandRequest):
    df, date, lag = make_df(features)
    forecast = sarimax_forecast(df, LAG_TEST, 'mse')
    return {date + datetime.timedelta(days=x): int(forecast[x]) for x in range(LAG_TEST)}

