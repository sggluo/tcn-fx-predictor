import numpy as np
from collections import defaultdict

import polars as pl
from polars import col as c

from src.configs.configs import (
    BASE_DIR, DATA_DIR, DATA_DIR_RAW, DATA_DIR_PROCESSED, DATA_DIR_FEATURES_TARGETS
)


def data_processing():
    lf = pl.scan_parquet(DATA_DIR_RAW/'raw_data.pqrquet').fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0)
    lf = lf.sort('timestamp').with_columns(
        c.timestamp.dt.truncate('10s'),
        micro_price = (c.askVolume * c.bidPrice + c.askPrice*c.bidVolume)/(c.bidVolume + c.askVolume)
          ).group_by('timestamp').agg(
              pl.all().last(),
              ask_min=c.askPrice.min(),
              bid_min=c.bidPrice.min(),
              ask_max=c.askPrice.max(),
              bid_max=c.bidPrice.max(),
              ask_std=c.askPrice.std(),
              bid_std=c.bidPrice.std()
              ).sort('timestamp')
    lf = lf.collect().upsample('timestamp', every='10s').with_columns(
        pl.all().exclude('timestamp').forward_fill()
    ).lazy()
    lf = lf.filter(c.timestamp.dt.weekday() <= 5)


    #————————————————————————————————————————
    lf = lf.with_columns(

        rolling_mean_10s = c.micro_price.rolling_mean(window_size=10),
        rolling_mean_30s = c.micro_price.rolling_mean(window_size=30),
        rolling_mean_60s = c.micro_price.rolling_mean(window_size=60),
        rolling_mean_180s = c.micro_price.rolling_mean(window_size=180),
    )
    lf = lf.with_columns(
        h10 = c.rolling_mean_10s.shift(-10),
        h30 = c.rolling_mean_30s.shift(-30),
        h60 = c.rolling_mean_60s.shift(-60),
        h180 = c.rolling_mean_180s.shift(-180),
    )

    lf = lf.with_columns(
        d10 = c.h10.log() - c.rolling_mean_10s.log(),
        d30 = c.h30.log() - c.rolling_mean_30s.log(),
        d60 = c.h60.log() - c.rolling_mean_60s.log(),
        d180 = c.h180.log() - c.rolling_mean_180s.log(),

    )
    # ——————————————-——————————————————————
    cols = [f'd{n}' for n in [10, 30, 60, 180]]
    lf = lf.drop_nulls()
    df = lf.collect()

    cutoff = int( df.height * 0.8 * 0.8 )
    df = df.slice(0,cutoff)
    quantiles = defaultdict(dict)

    for cl in cols:
        quantiles[cl]['.01_q'] = df.select(cl).quantile(0.01).item()
        quantiles[cl]['.4_q'] = df.select(cl).quantile(0.4).item()
        quantiles[cl]['.6_q'] = df.select(cl).quantile(0.6).item()
        quantiles[cl]['.99_q'] = df.select(cl).quantile(0.99).item()


    #——————————————————————————————————
    lf = lf.with_columns(
        [ pl.col(cl).clip(quantiles[cl]['.01_q'], quantiles[cl]['.99_q'] ) for cl in cols
          ]
    )
    thrd = defaultdict(dict)
    for cl in cols:
        thrd[cl]['threshold'] = lf.select( pl.col(cl).abs()).quantile(0.1).collect().item()

    lf = lf.with_columns(
        [ (pl.when( pl.col(cl)>thrd[cl]['threshold'] ).then(2).when(pl.col(cl)<-1*thrd[cl]['threshold']).then(0).otherwise(1)).alias(cl) for cl in cols
          ]
    )
    cols2drop = [f'h{n}' for n in [10,30,60,180]] +[f'rolling_mean_{n}s' for n in [10,30,60,180]]
    lf = lf.drop(cols2drop)
    lf.sink_parquet(DATA_DIR_PROCESSED/'dataset_with_targets.parquet')
    return lf

def sliding_window(X, sql):
  X_list = []
  for i in range(X.shape[0]-sql+1):
      X_list.append(X[i:i+sql, :])
  X = np.stack(X_list, axis=0)
  return X


def feature_engineering():
    data = pl.scan_parquet(DATA_DIR_PROCESSED/'dataset_with_targets.parquet').sort('timestamp')
    data = data.with_columns(
        order_imbalance = c.bidVolume.log()-c.askVolume.log(),
        total_depth = c.bidVolume + c.askVolume,
        delta_bid_volume = c.bidVolume.diff(),
        delta_ask_volume = c.askVolume.diff(),
        micro_ret = c.micro_price.log() - c.micro_price.shift(1).log()
)

    data =data.with_columns(
        rolling_ret_mean_5 = c.micro_ret.rolling_mean(window_size=5),
        rolling_ret_mean_10 = c.micro_ret.rolling_mean(window_size=10),
        rolling_ret_mean_50 = c.micro_ret.rolling_mean(window_size=50),
)  
    data = data.fill_null(strategy='backward')
    return data

def  train_test_split(data, train_ratio=0.8, val_ratio=0.2):
    target_cols = [f'd{n}' for n in [10,30,60,180]]
    X = data.select(pl.all().exclude(target_cols+['timestamp'])).collect().to_numpy()
    y = data.select(target_cols).collect().to_numpy()
    cutoff1 = int( y.shape[0] * train_ratio * (1-val_ratio) )
    cutoff2 = int( y.shape[0] * train_ratio )
    X_train = X[:cutoff1]
    y_train = y[:cutoff1]
    X_val = X[cutoff1:cutoff2][180:]
    y_val = y[cutoff1:cutoff2][180:]
    X_test = X[cutoff2:][180:]
    y_test = y[cutoff2:][180:]
    return X_train,y_train, X_val, y_val, X_test, y_test






