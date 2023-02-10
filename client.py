import requests
import json

with open('data/2022_12.txt') as data_source:
    data_requests = data_source.readlines()

MODELS = ['lstm', 'etna', 'sarimax/aic', 'sarimax/mse']

results = dict()
for data_request in data_requests:
    data = json.loads(data_request)
    forecasts, date = dict(), data['current_date']
    for model in MODELS:
        response = requests.post(f'http://0.0.0.0:9035/predicts/{model}', json=data)
        forecasts[model] = response.json()
    print(forecasts)
    results[date] = forecasts

with open('december.json', 'w') as fp:
    json.dump(results, fp)
