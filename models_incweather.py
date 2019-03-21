import csv
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import holidays
from keras import models, layers, callbacks
import os
import statsmodels.tsa.api as smt
import seaborn as sns
import matplotlib.gridspec as gs
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import pickle

from helpers import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# data processing step #############################
# import processed data; if using other profiles, run section in profile_proc.py and continue below ###############
aggregate_load = pd.DataFrame()
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)

# source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load = aggregate_load.apply(lambda x: x/1000)
aggregate_load.index = pd.to_datetime(aggregate_load.index)

# duplicate index due to additional hour in Nov due to DST
joined = aggregate_load.groupby(aggregate_load.index).first()
joined = joined.dropna().copy()

# Weather
with open('houston_weather.csv', 'r') as f:
    weather = pd.read_csv(f, index_col=0)

joined = aggregate_load.join(weather, how='inner')  # joining on index
joined = joined.groupby(joined.index).first()  # duplicate index due to additional hour in Nov due to DST
joined = joined[['COAST', 'Drybulb', 'Humidity']].dropna().copy()

# create indicator variables
joined['Hour_Num'] = joined.index.hour
joined['Day_Num'] = joined.index.weekday  # Monday is 0
joined['Wknd_Flag'] = (joined.index.weekday > 4) * 1
joined['Date'] = joined.index.date
joined['Month'] = joined.index.month
joined['Day'] = joined.index.day
joined['Day_Year'] = joined.index.dayofyear
joined['Week_Year'] = joined.index.weekofyear

us_holidays = holidays.UnitedStates()  # this creates a dictionary
joined['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined['Date']]
# to explicitly use the lambda function for the same effect as above
# joined['Holiday_Flag'] = [(lambda x: (x in us_holidays) * 1)(x) for x in joined['Date']]
joined['Off_Flag'] = joined[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)
joined = joined.drop(columns=['Date', 'Wknd_Flag', 'Holiday_Flag'])
joined = joined.dropna()
int_cols = ['Hour_Num', 'Day_Num', 'Day', 'Month', 'Day_Year', 'Week_Year', 'Off_Flag', 'Drybulb', 'Humidity']
joined[int_cols] = joined[int_cols].applymap(lambda x: int(x))

# create training and testing data sets, generator will separate out the features and target
train_data = joined[joined.index.year != 2017]
test_data = pd.concat([train_data[-1440:],
                      joined[joined.index.year == 2017]])
# test_data = joined[joined.index.year == 2017]

# added section for standardization
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# create generators using udf in helpers file
lookback = 1440  # 60 days
delay = 24  # how far into the future is the target

# new generator function for samples and targets
# move the target variable (regional load to predict) to the first column position
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batchsize=168, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batchsize)
        else:
            if i + batchsize >= max_index:
                i = min_index + lookback
            # np.arange(start, stop(not including))
            rows = np.arange(i, min(i + batchsize, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback//step,
                           data.shape[-1]))
        targets = np.zeros(len(rows), )
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets


train_gen = generator(train_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=50000)
val_gen = generator(train_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=50001,
                    max_index=None)

# RNN model
def build_rnn():
    model = models.Sequential()
    model.add(layers.GRU(64, input_shape=(None, train_data.shape[-1])))
    # model.add(layers.GRU(64, input_shape=(None, 1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model_rnn = build_rnn()
history_rnn = model_rnn.fit_generator(train_gen,
                                      steps_per_epoch=365,
                                      epochs=20,
                                      validation_data=val_gen,
                                      validation_steps=50,
                                      callbacks=[
                                          callbacks.ReduceLROnPlateau(factor=.5, patience=3, verbose=1)
                                      ]
                                      )


# with open(f'models/rnn_20_wd_ww.pickle', 'wb') as pfile:
#     pickle.dump(model_rnn, pfile)
# with open(f'models/rnn_20_wd_ww_hist.pickle', 'wb') as pfile:
#     pickle.dump(history_rnn, pfile)

with open(f"models/rnn_20_wd_ww.pickle", "rb") as pfile:
    exec(f"model_rnn = pickle.load(pfile)")
with open(f"models/rnn_20_wd_ww_hist.pickle", "rb") as pfile:
    exec(f"history_rnn = pickle.load(pfile)")

# loss_plot(history=history_rnn, skip_epoch=0)

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

test_metrics_rnn = model_rnn.evaluate_generator(test_gen, steps=1)
test_mae = test_metrics_rnn[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[1440:(1440+168*1), 0].mean()))
# test accuracy: 0.86595 (1 step, with dummies and weather)
# test accuracy: 0.85863 (3 step, with dummies and weather)
# test accuracy: 0.78685 (10 step, with dummies and weather)

# shifted
# test accuracy: 0.77193 (1 step, with dummies and weather)
# test accuracy: 0.84380 (3 step, with dummies and weather)
# test accuracy: 0.82917 (10 step, with dummies and weather)

pred_plot_per_step2(test_data, model_crnn)

pred_multiplot2(model_rnn, test_data)

# Dense model
def build_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback, train_data.shape[-1])))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(40, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit_generator(train_gen,
                              epochs=10,
                              steps_per_epoch=365,
                              validation_data=val_gen,
                              validation_steps=50,
                              callbacks=[
        callbacks.ReduceLROnPlateau(factor=.5, patience=3, verbose=1)
    ])


with open(f"models/dense_20_wd_ww.pickle", "rb") as pfile:
    exec(f"model = pickle.load(pfile)")
with open(f"models/dense_20_wd_ww_hist.pickle", "rb") as pfile:
    exec(f"history = pickle.load(pfile)")

loss_plot(history=history, skip_epoch=0)

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

test_metrics = model.evaluate_generator(test_gen, steps=10)
test_mae = test_metrics[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[:168*10, 0].mean()))
# test accuracy: 0.85002 (1 step, with dummies and weather)
# test accuracy: 0.78907 (3 step, with dummies and weather)
# test accuracy: 0.75699 (10 step, with dummies and weather)

pred_multiplot2(model, test_data)

with open(f"models/crnn_20_wd_ww.pickle", "rb") as pfile:
    exec(f"model_crnn = pickle.load(pfile)")
with open(f"models/crnn_20_wd_ww_hist.pickle", "rb") as pfile:
    exec(f"history_crnn = pickle.load(pfile)")

# loss_plot(history=history_crnn, skip_epoch=0)

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

test_metrics_crnn = model_crnn.evaluate_generator(test_gen, steps=10)
test_mae = test_metrics_crnn[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[1440:(1440+168*10), 0].mean()))

# test accuracy: 0.86522 (1 step, with dummies and weather, cnn + rnn)
# test accuracy: 0.83156 (3 step, with dummies and weather, cnn + rnn)
# test accuracy: 0.76871 (10 step, with dummies and weather, cnn + rnn)

# shifted
# test accuracy: 0.77024 (1 step, with dummies and weather, cnn + rnn)
# test accuracy: 0.83755 (3 step, with dummies and weather, cnn + rnn)
# test accuracy: 0.83747(10 step, with dummies and weather, cnn + rnn)

pred_multiplot2(model_crnn, test_data)

pred_plot_per_step2(test_data, model_crnn)

