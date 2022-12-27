from datetime import date

from pydantic import BaseModel, conlist, root_validator


MIN_LEN = 240

class RegressorsSection(BaseModel):
    """
    Minimum regressors length is 31*2 elements. Maximum length is 31*3
    """
    orders_count: conlist(int, min_items=MIN_LEN)
    new_orders_count: conlist(int, min_items=MIN_LEN)
    custom_orders_rate: conlist(int, min_items=MIN_LEN)
    orders_count_last_year: conlist(int, min_items=MIN_LEN)

    @root_validator
    def local_domain_len(cls, values):
        keys = [
            'orders_count',
            'new_orders_count',
            'custom_orders_rate',
            'orders_count_last_year'
        ]
        is_same_len = lambda l: l == len(values[keys[0]])
        assert all(map(is_same_len, [len(values[key]) for key in keys])), \
                'all regressors should be same length'
        return values


class DemandRequest(BaseModel):
    regressors: RegressorsSection
    current_date: date
    # predict_days_n: int
