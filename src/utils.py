import math
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay

def get_last_business_day(date_param):
    date = pd.to_datetime(date_param) if date_param else datetime.now()
    last_business_day = date - BDay(1)
    return last_business_day.date()

def accuracy_confidence_interval(acc, n, z=1.96):
    # acc: measured accuracy (e.g., 0.53)
    # n: number of samples
    se = math.sqrt((acc * (1 - acc)) / n)
    lower = acc - z * se
    upper = acc + z * se
    return lower, upper