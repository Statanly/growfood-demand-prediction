import subprocess
from multiprocessing import Pool
from tqdm import tqdm


def run(params: list[str]):
    product_name, city_id, until_date = params
    cmd = f'python3 ./experiment_121222.py {product_name} {city_id} {until_date}'.split()
    subprocess.run(cmd)


if __name__ == '__main__':
    params = [
        ['balance', 1],
        ['basic', 1],
        ['breakfasts_2x', 1],
        ['daily', 1],
        ['detox', 1],
        # ['fit', 1],
        ['fit_express', 1],
        ['m_fit', 1],
        ['power', 1],
        ['priem', 1],
        ['priem_plus', 1],
        ['super_fit', 1],

        ['balance', 2],
        ['basic', 2],
        ['breakfasts_2x', 2],
        ['daily', 2],
        ['detox', 2],
        ['fit', 2],
        ['fit_express', 2],
        ['m_fit', 2],
        ['power', 2],
        ['priem', 2],
        # ['priem_plus', 2],
        ['super_fit', 2]
    ]

    until_dates = [
        '2022-11-01',
        '2022-10-01',
        '2022-09-01',
        '2022-08-01',
        '2022-07-01',
        '2022-06-01',
        '2022-05-01',
        '2022-04-01',
        '2022-03-01',
        '2022-02-01',
        '2022-01-01',
        '2021-12-01',
    ]
    new_params = []
    for p in params:
        for date in until_dates:
            new_params.append((p[0], p[1], date))

    with Pool(min(64, len(params))) as pool:
        list(tqdm(pool.imap(run, new_params)))
