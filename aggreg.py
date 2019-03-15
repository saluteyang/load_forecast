import pandas as pd
import numpy as np
import holidays
from keras import models, layers, callbacks
import os
from sklearn import preprocessing
import pickle

import seaborn as sns
sns.set()

from helpers import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# data processing step #############################
# import processed data; if using other profiles, run section in profile_proc.py and continue below ###############
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)

# source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load = aggregate_load.apply(lambda x: x/1000)
aggregate_load.index = pd.to_datetime(aggregate_load.index)

# duplicate index due to additional hour in Nov due to DST
joined = aggregate_load.groupby(aggregate_load.index).first()
joined = joined.dropna().copy()

##################################
# specify the region to forecast
region = 'COAST'
cols = [x for x in joined.columns if x!=region]  # get region columns to be dropped later

# create indicator variables
joined['Hour_Num'] = joined.index.hour
joined['Day_Num'] = joined.index.weekday  # Monday is 0
joined['Wknd_Flag'] = (joined.index.weekday > 4) * 1
joined['Date'] = joined.index.date
joined['Month'] = joined.index.month
joined['Day'] = joined.index.day
joined['Day_Year'] = joined.index.dayofyear
joined['Week_Year'] = joined.index.weekofyear

# add holidays flag
us_holidays = holidays.UnitedStates()  # this creates a dictionary
joined['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined['Date']]
joined['Off_Flag'] = joined[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)

# create lagged variables
joined = joined.merge(joined.groupby(joined.index.date)['COAST'].mean().to_frame(),
                                left_on='Date', right_index=True,
                                suffixes=['_Hourly', '_DailyAve'])  # syntax different from join method
joined['COAST_Hourly_Pre_Day'] = joined['COAST_Hourly'].shift(1, 'd').to_frame()
joined['COAST_Hourly_Pre_Wk_Day'] = joined['COAST_Hourly'].shift(7, 'd').to_frame()
joined['COAST_DailyAve_Pre_Day'] = joined['COAST_DailyAve'].shift(1, 'd').to_frame()

joined = joined.drop(columns=cols + ['Date', 'Wknd_Flag', 'Holiday_Flag'])
joined = joined.dropna()

# hold out 2017 data
train_data = joined[joined.index.year != 2017].drop(columns='COAST_Hourly')
train_target = joined[joined.index.year != 2017]['COAST_Hourly']
test_data = joined[joined.index.year == 2017].drop(columns='COAST_Hourly')
test_target = joined[joined.index.year == 2017]['COAST_Hourly']

# added section for standardization
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# all but the last 10000 used for training, val is the rest
partial_train_data = train_data[:-10000]
val_data = train_data[-10000:]
partial_train_target = train_target[:-10000]
val_target = train_target[-10000:]

partial_train_target = partial_train_target.values[336:]
test_target = test_target.values[336:]
val_target = val_target.values[336:]


# redimension datasets for RNN
def rnn_redim(data, lookback=336):
    # rows are from 336 to number of rows in the data passed in
    rows = np.arange(lookback, data.shape[0])
    # dims of samples: data row number - lookback, lookback, number of features
    samples = np.zeros((len(rows), lookback, data.shape[1]))
    # for first iteration below: j=0, row=336 (row is not used)
    # for second iteration: j=1, row=337
    for j, row in enumerate(rows):
        # for first iteration: indices = range(0, 336)
        # for second iteration: indices = range(1, 337)
        indices = range(rows[j] - lookback, rows[j])
        # for first iteration: fill samples[0], a lookback x features matrix
        # with data[range(0, 336)]
        samples[j] = data[indices]  # sliding lookback window through the data
    return samples

partial_train_data_3d = rnn_redim(partial_train_data)
val_data_3d = rnn_redim(val_data)
test_data_3d = rnn_redim(test_data)

def build_model_corrected():
    model = models.Sequential()
    model.add(layers.GRU(64, input_shape=(None, train_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model_corrected = build_model_corrected()
history = model_corrected.fit(partial_train_data_3d,
                              partial_train_target,
                              epochs=20, batch_size=168,
                              validation_data=(val_data_3d,
                                               val_target))
