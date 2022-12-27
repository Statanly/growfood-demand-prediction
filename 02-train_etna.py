#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


# # load data

# In[2]:


root = Path('data/processed/')


# In[3]:


orders_count = pd.read_csv(root / 'orders_count.csv')
orders_count.head(3)


# In[4]:


new_orders_count = pd.read_csv(root / 'new_orders_count.csv')
new_orders_count.head(3)


# In[5]:


fact_deliveries_count = pd.read_csv(root / 'fact_deliveries_count.csv')
fact_deliveries_count.head(3)


# In[6]:


f = lambda s: s.split('_')[0]
fact_deliveries_count = fact_deliveries_count.assign(product_name=fact_deliveries_count.name.apply(f))
fact_deliveries_count.head(3)


# # make dataset

# In[7]:


city_id = 1
product = 'balance'

df1 = orders_count[(orders_count.city_id == city_id) & (orders_count.product_name == product)]
df2 = new_orders_count[(new_orders_count.city_id == city_id) & (new_orders_count.product_name == product)]
df3 = fact_deliveries_count[(fact_deliveries_count.city_id == city_id) & 
                            (fact_deliveries_count.product_name == product)] \
            .rename({'planned_delivery_date': 'date'}, axis=1) \
            .drop_duplicates('date')
cols = ['date', 'count']

df = pd.merge(df1[cols], df2[cols], on='date')
df = df.rename(dict(count_x='orders_count',
                    count_y='new_orders_count'), axis=1)
df = pd.merge(df, df3[cols], on='date').rename(dict(count='deliveries_count'), axis=1)
df.sort_values('date', inplace=True)
df.head(3)


# In[8]:


def fill_nan_dates(subdf: pd.DataFrame) -> pd.DataFrame:
    d = subdf.copy()
    d.index = pd.to_datetime(d.date)
    d = d.drop('date', axis=1)

    d = d.asfreq('1d')
    d.fillna(method='ffill', inplace=True)
    d = d.reset_index()
    return d

df = fill_nan_dates(df)
df = df.iloc[-200:]


# In[9]:


df = df.rename(dict(date='timestamp', deliveries_count='target'), axis=1)        .assign(segment='main')
df.head()


# In[10]:


df_raw = df.copy()


# In[11]:


df_raw.head()


# # make etna dataset

# In[12]:


from etna import transforms as T


# In[13]:


target_transforms = [
    T.MeanTransform(in_column='target', window=7, out_column='mean_3'),
    
    T.FourierTransform(period=7, order=2, out_column='fourier_7_2'),
    T.FourierTransform(period=9, order=2, out_column='fourier_9_2'),
    
    T.DifferencingTransform(in_column='target', period=2, order=1, out_column='diff_2_1'),
    T.DifferencingTransform(in_column='target', period=2, order=2, out_column='diff_2_2'),
    
    T.HolidayTransform(iso_code='RUS', out_column='holiday_ru'),
]

exog_transforms = [
    T.LagTransform(in_column='orders_count', lags=[5, 6, 7], out_column='lag_orders_count'),
    T.LagTransform(in_column='new_orders_count', lags=[2, 7, 9], out_column='lag_new_orders_count')
]


# In[14]:


import pandas as pd
from etna.datasets import TSDataset

exog_features = df_raw.drop(['target'], axis=1)
exog_ts = TSDataset.to_dataset(exog_features)

# apply transforms
for tr in exog_transforms:
    exog_ts = tr.fit_transform(exog_ts)
exog_ts = exog_ts.fillna(0)


# Create a TSDataset
cols = [
    'timestamp', 
    'target', 
    'segment'
]
df = TSDataset.to_dataset(df_raw[cols].iloc[:-1])
# apply transforms
for tr in target_transforms:
    df = tr.fit_transform(df)
ts = TSDataset(df, freq="D", df_exog=exog_ts, known_future='all')

# # Choose a horizon
# HORIZON = 14

# # Make train/test split
# train_ts, test_ts = ts.train_test_split(test_size=HORIZON)


# In[15]:


ts.df = ts.df.fillna(0)
ts.df_exog = ts.df_exog.fillna(0)
ts.head(3)


# In[16]:


ts.regressors


# In[17]:


ts.plot()


# In[18]:


from etna.analysis import sample_pacf_plot

sample_pacf_plot(ts, lags=14)


# In[19]:


from etna.analysis.outliers import get_anomalies_density
from etna.analysis import plot_anomalies

anomaly_seq_dict = get_anomalies_density(
  ts, window_size=31, distance_coef=1, n_neighbors=10)
plot_anomalies(ts, anomaly_seq_dict)


# In[ ]:





# # one model prediction

# In[20]:


from etna.models.nn.mlp import MLPModel
from etna.transforms import PytorchForecastingTransform, DateFlagsTransform

import torch
import numpy as np
import random 

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# In[21]:


_, cols = zip(*ts.df.columns.tolist())
cols = list(cols)

real_cols = list(cols)
for col in (
    'holiday_ru', 
    'target'
):
    real_cols.remove(col)
cat_cols = ['holiday_ru']


# In[22]:


from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

HORIZON = 7

transform_date = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="dateflag")

transform_deepar = PytorchForecastingTransform(
    max_encoder_length=HORIZON,
    max_prediction_length=HORIZON,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["target"] + real_cols,
    # time_varying_known_categoricals=["dateflag_day_number_in_week"] + cat_cols,
    time_varying_unknown_categoricals=cat_cols,
    # categorical_encoders=dict(
    #     holiday_ru=NaNLabelEncoder(add_nan=True)
    # ),
    target_normalizer=GroupNormalizer(groups=["segment"]),
)


# In[23]:


# from etna.settings import SETTINGS

# SETTINGS.torch_required


# In[24]:


from etna.models.nn.deepar import DeepARModel
from etna.metrics import SMAPE, MAPE, MAE
from etna.pipeline import Pipeline

# mlp_params = dict(
#     input_size=10,
#     decoder_length=HORIZON,
#     hidden_size=32,
#     encoder_length=HORIZON,
#     train_batch_size=64,
#     test_batch_size=64
# )


model_deepar = DeepARModel(context_length=HORIZON, max_epochs=5, learning_rate=[0.01], gpus=0, batch_size=64)
metrics = [SMAPE(), MAPE(), MAE()]

pipeline_deepar = Pipeline(
    model=model_deepar,
    horizon=HORIZON,
    transforms=[transform_date, transform_deepar],
)


# In[25]:


metrics_deepar, forecast_deepar, fold_info_deepar = pipeline_deepar.backtest(ts, metrics=metrics, n_folds=1, n_jobs=1)


# In[ ]:


ts


# In[ ]:


ts


# In[ ]:


metrics_deepar


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from etna.models import (
    SeasonalMovingAverageModel,
    LinearPerSegmentModel,
    CatBoostPerSegmentModel,
    HoltWintersModel
) 
from etna.ensembles import VotingEnsemble
from etna.pipeline import Pipeline
from etna.transforms import BinsegTrendTransform
from sklearn.ensemble import GradientBoostingRegressor


HORIZON = 1
pipelines = [
    # Pipeline(
    #     model=CatBoostPerSegmentModel(),
    #   # model=SeasonalMovingAverageModel(window=10, seasonality=7),
    #   transforms=[
    #     # BinsegTrendTransform(
    #     #   in_column="target",
    #     #   min_size=12,
    #     #   jump=1,
    #     #   model="ar",
    #     #   n_bkps=10
    #     # ),
    #   ],
    #   horizon=HORIZON
    # ),
    # Pipeline(
    #     model=SeasonalMovingAverageModel(window=20, seasonality=7),
    #     horizon=HORIZON
    # ),
    # Pipeline(
    #     model=HoltWintersModel(use_boxcox=True, trend='add', damped_trend=False, seasonal='add', seasonal_periods=7),
    #     horizon=HORIZON
    # ),
    # Pipeline(
    #     model=HoltWintersModel(use_boxcox=True, trend='add', damped_trend=False, seasonal='add', seasonal_periods=7),
    #     horizon=HORIZON
    # )
]

# voting_ensemble = VotingEnsemble(pipelines, regressor=GradientBoostingRegressor)


# In[ ]:


from etna.metrics import MAE, SMAPE, MSE

metrics_list = [MAE(), SMAPE(), MSE()]
metrics_df, backtest_df, _ = pipelines[0].backtest(
    ts=ts,
    metrics=metrics_list,
    n_folds=20,
    aggregate_metrics=True
)


# In[ ]:


metrics_df


# In[ ]:


from etna.analysis.plotters import plot_backtest

plot_backtest(backtest_df, ts, history_len=20)


# In[ ]:





# In[ ]:





# # Other models

# In[ ]:


Pipeline(
        model=HoltWintersModel(use_boxcox=True, trend='add', damped_trend=False, seasonal='add', seasonal_periods=7),
        horizon=HORIZON
    )


# In[ ]:





# In[ ]:





# In[ ]:




