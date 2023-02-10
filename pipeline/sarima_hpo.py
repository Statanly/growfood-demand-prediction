import datetime
import itertools
import warnings

import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from pipeline.src.transforms import WeekdayTransform

warnings.filterwarnings("ignore")


def optimizeSARIMA(x, exog, parameters_list, d, D):
    """Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in tqdm(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            p, q, P, Q, s = param
            model = sm.tsa.statespace.SARIMAX(
                x, order=(p, d, q), seasonal_order=(P, D, Q, s),
                trend='ct'
            ).fit(disp=-1, maxiter=70)
        except Exception as e:
            print('err', e)
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic, model.mse])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic', 'mse']
    return result_table


raw = [517, 446, 760, 427, 713, 432, 686, 408, 460, 670, 486, 638, 504, 591, 453, 372, 685, 383, 645, 389, 632, 359, 440,
      633, 455, 585, 457, 581, 447, 385, 684, 403, 676, 415, 637, 379, 426, 640, 455, 597, 461, 572, 420, 344, 650, 360,
      622, 355, 577, 331, 360, 385, 364, 442, 366, 432, 348, 305, 387, 306, 528, 346, 526, 341, 397, 572, 422, 533, 424,
      499, 367, 334, 587, 351, 587, 337, 550, 299, 351, 540, 358, 497, 338, 480, 326, 290, 532, 315, 517, 315, 482, 276,
      291, 371, 402, 402, 383, 404, 357, 268, 525, 279, 520, 296, 479, 252, 299, 479, 310, 484, 310, 430, 275, 214, 495,
      250, 466, 272, 435, 242, 261, 443, 298, 443, 311, 405, 271, 237, 467, 275, 454, 276, 421, 239, 246, 431, 283, 424,
      280, 411, 256, 278, 430, 306, 425, 285, 401, 263, 253, 426, 305, 422, 299, 409, 281, 240, 487, 284, 458, 284, 429,
      259, 268, 474, 325, 458, 305, 441, 279, 277, 471, 333, 482, 321, 466, 315, 311, 527, 371, 538, 365, 487, 329, 297,
      558, 367, 560, 369, 529, 356, 354, 583, 404, 576, 361, 546, 327, 324, 564, 340, 543, 327, 483, 308, 304, 508, 338,
      504, 336, 477, 298, 279, 529, 328, 508, 322, 477, 361, 331, 550, 389, 552, 403, 496, 348, 328, 564, 334, 533, 328,
      477, 303, 304, 495, 350, 479, 325, 408, 276, 278]

date = datetime.datetime(2022, 11, 6)
df = pd.DataFrame()
df.loc[:, 'deliveries_count'] = raw
dates = [date - datetime.timedelta(days=x) for x in range(len(df), 0, -1)]
df.loc[:, 'date'] = dates
df = WeekdayTransform(name='weekday', column='date').transform(df)

# buckets = [list() for _ in range(7)]
# for target, d in zip(df, dates):
#     buckets[d.weekday()].append(float(target))
# means = [np.mean(bucket) for bucket in buckets]
weekdays = [f'weekday__date_{idx}' for idx in range(7)]
exog = df[weekdays]
ps = range(3, 5)
dp=1
qs = range(1, 3)
Ps = range(0, 2)
D=1
Qs = range(0, 2)
ss = [7, 14, 30]

# creating list with all the possible combinations of parameters
parameters = itertools.product(ps, qs, Ps, Qs, ss)
parameters_list = list(parameters)

result_table = optimizeSARIMA(df['deliveries_count'], exog, parameters_list, dp, D)
# sorting in ascending order, the lower AIC is - the better
aic = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
mse = result_table.sort_values(by='mse', ascending=True).reset_index(drop=True)
print(aic.head(5))
print(mse.head(5))


# def plotSARIMA(series, model, n_steps):
#     """Plots model vs predicted values
#
#         series - dataset with timeseries
#         model - fitted SARIMA model
#         n_steps - number of steps to predict in the future
#     """
#
#     # adding model values
#     data = pd.DataFrame(series.copy())
#     data.columns = ['actual']
#     data = data.assign(sarima_model=model.fittedvalues)
#     # making a shift on s+d steps, because these values were unobserved by the model
#     # due to the differentiating
#     data['sarima_model'][:s + dp] = np.NaN
#
#     # forecasting on n_steps forward
#     forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
#     forecast = data.sarima_model.append(pd.Series(forecast))
#     # calculate error, again having shifted on s+d steps from the beginning
#     error = mean_absolute_percentage_error(data['actual'][s + dp:], data['sarima_model'][s + dp:])
#
#     plt.figure(figsize=(25, 7))
#     plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
#     plt.plot(forecast, color='r', label="model")
#     plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
#     plt.plot(data.actual, label="actual")
#     plt.legend()
#     plt.grid(True)


fore_dates = [date + datetime.timedelta(days=x) for x in range(10)]
fore = pd.DataFrame()
fore.loc[:, 'date'] = fore_dates
fore = WeekdayTransform(name='weekday', column='date').transform(fore)


def eval_model(p, q, P, Q, s):
    best_model = sm.tsa.statespace.SARIMAX(
        df['deliveries_count'], exog=exog, order=(p, dp, q), seasonal_order=(P, D, Q, s), trend='ct'
    ).fit(disp=-1)
    print(best_model.forecast(10))


eval_model(3, 1, 1, 0, 30)
eval_model(4, 2, 0, 0, 7)
# [518, 314, 484, 354, 474, 329, 317]


# (3, 1, 1, 0, 30) aic exog
# (4, 2, 0, 0, 7) mse exog


# (4, 2, 1, 1, 30) aic
# (4, 2, 0, 0, 7) mse

