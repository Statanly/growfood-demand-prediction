import datetime

from fastapi import APIRouter
import pandas as pd

from src.predictor import Predictor
from src.demand_request import DemandRequest


predict_router = APIRouter()
predictor = Predictor()


def make_df(features: DemandRequest) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(dict(features.regressors))

    date = features.current_date
    dates = [date - datetime.timedelta(days=x) for x in range(len(df))]
    df.loc[:, 'date'] = dates

    df.rename(columns={'deliveries_count': 'target'}, inplace=True)

    # lag depends on city
    if features.city_id == 1:
        # spb
        lag = 3
    elif features.city_id == 2:
        # msk
        lag = 4
    else:
        raise ValueError('Unknown city_id')

    lagged_target = df.target.iloc[lag:].tolist()
    df = df.iloc[:-lag]
    df.loc[:, 'target'] = lagged_target

    # rearrange columns
    cols_order = [
        'date',
        'target',
        'new_orders_count',
        'custom_orders_rate',
    ]
    df = df[cols_order]
    return df


@predict_router.post("/predict")
async def predict(features: DemandRequest):
    df = make_df(features)
    res = predictor.fit_predict(df)

    return res
