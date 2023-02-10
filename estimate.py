import json

import numpy as np
from dateutil import parser

with open('merged.json') as fp:
    results = json.load(fp)
with open('data/true.json') as fp:
    targets = json.load(fp)

mape_dict = {
    model: [list() for lag in range(5)]
    for model in ['lstm', 'etna', 'sarimax/aic', 'sarimax/mse']
}

for req_date, models in results.items():
    for model, forecasts in models.items():
        for fore_date, fore_target in forecasts.items():
            true_target = targets[fore_date]
            if true_target < 100:
                continue
            lag = (parser.parse(fore_date) - parser.parse(req_date)).days
            mape = abs(fore_target - true_target) / abs(true_target)
            mape_dict[model][lag].append(mape)

for model, lags in mape_dict.items():
    print(f'- {model} -')
    print([np.around(np.mean(mape_list), 4) for mape_list in lags])

