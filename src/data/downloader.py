import datetime as dt
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD

import polars as pl
from polars import col as c
import pandas as pd
from src.configs.configs import DATA_DIR_RAW, DATA_DIR_PROCESSED, DATA_DIR_FEATURES_TARGETS

def download(start='2025-01-02', end='2025-01-31'):
    trading_days = pd.bdate_range(start=start, end=end)
    last_day = trading_days[-1] + pd.Timedelta(days=1)
    trading_days = trading_days.union(pd.DatetimeIndex([last_day]))

    s = 0
    data_list=[]
    while True:
        e = min( s+1, len(trading_days)-1)
        start = trading_days[s]
        end = trading_days[e]

        data = dukascopy_python.fetch(
            instrument=INSTRUMENT_FX_MAJORS_EUR_USD,
            interval=dukascopy_python.INTERVAL_TICK,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=start,
            end=end,
)
        if not data.empty:
            data = data.reset_index()
            data_list.append( pl.from_pandas(data).lazy() )
        s = e
        if e == len(trading_days)-1:
            break


    lf = pl.concat(data_list, how='vertical')
    lf.sink_parquet(DATA_DIR_RAW/'raw_data.pqrquet')
    return lf
