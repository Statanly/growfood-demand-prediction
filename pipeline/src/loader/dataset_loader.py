from pathlib import Path

import pandas as pd


class DatasetLoader:
    def __init__(self, root: Path, product_name: str, city_id: int, until_date: str = '2022-12-20'):
        self._df = self.__load_files(root, product_name, city_id, until_date)
        self._df = self.__fill_nan_dates(self._df)

    @staticmethod
    def __load_files(root: Path, product_name: str, city_id: int, until_date: str) -> pd.DataFrame:
        tnames = [
            'orders_count',
            'new_orders_count',
            'custom_orders_rate',
            # 'discounts',
            # 'boxes_per_delivery',
            # 'fooddays_per_order'
        ]

        for table in tnames:
            t = pd.read_csv(root / f'{table}.csv')
            t = t.rename(columns={'count': table})
            t = t[(t.city_id == city_id) & (t.product_name == product_name)]
            t = t[['date', table]]

            if table == 'orders_count':
                orders_count = t.rename(columns={'orders_count': 'target'})
            else:
                orders_count = pd.merge(orders_count, t, on='date', how='left').fillna(0)

        orders_count.sort_values('date', inplace=True)

        # cut by date
        orders_count = orders_count[orders_count.date < until_date]
        print('df length', len(orders_count))
        # cut last n views
        orders_count = orders_count.iloc[-240:]
        print('df length', len(orders_count))

        return orders_count

    @staticmethod
    def __fill_nan_dates(df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.date)
        df = df.drop('date', axis=1)
        df = df.asfreq('1d')
        df.fillna(method='ffill', inplace=True)
        df = df.reset_index()

        return df

    def get_dataset(self) -> pd.DataFrame:
        return self._df
