import csv
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import holidays
from keras import models
from keras import layers
from keras import callbacks
# import keras
import os
import statsmodels.tsa.api as smt
import seaborn as sns
import matplotlib.gridspec as gs
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import numpy as np

from models_noweather_helpers import *

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

# create indicator variables
joined['Hour_Num'] = joined.index.hour
joined['Day_Num'] = joined.index.weekday  # Monday is 0
joined['Wknd_Flag'] = (joined.index.weekday > 4) * 1
joined['Date'] = joined.index.date

# add holidays flag
us_holidays = holidays.UnitedStates()  # this creates a dictionary
joined['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined['Date']]
joined['Off_Flag'] = joined[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)
joined = joined.drop(columns=['Date', 'Wknd_Flag', 'Holiday_Flag'])

# create training and testing data sets, generator will separate out the features and target
train_data = joined[joined.index.year != 2017]
test_data = joined[joined.index.year == 2017]

# added section for standardization
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# create generators using udf in helpers file
lookback = 1440  # 60 days
delay = 24  # how far into the future is the target

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
test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

# val_steps = (60000 - 50001 - lookback)
# test_steps = (len(joined) - 60001 - lookback)

# regression model  #########################################################################################

def build_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback, joined.shape[-1])))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit_generator(train_gen,
                              epochs=20,
                              steps_per_epoch=100,
                              validation_data=val_gen,
                              validation_steps=100)

test_metrics = model.evaluate_generator(test_gen, steps=24)
test_mae = test_metrics[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[:24, 0].mean()))
# test accuracy: 0.45951

loss_pred_plots(history=history, skip_epoch=0, model=model,
                test=test_gen, test_target=test_data[:, 0], pred_periods=48)

# the RNN models ####################################################################

# early stopping?
def build_rnn():
    model = models.Sequential()
    model.add(layers.GRU(64, input_shape=(None, train_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model_rnn = build_rnn()
history_rnn = model_rnn.fit_generator(train_gen,
                                      steps_per_epoch=100,
                                      epochs=20,
                                      validation_data=val_gen,
                                      validation_steps=100
                                      # callbacks=[
                                      #     callbacks.EarlyStopping(patience=5, verbose=1,
                                      #                             restore_best_weights=True)
                                      # ]
                                      )


test_metrics_rnn = model_rnn.evaluate_generator(test_gen, steps=24)
test_mae = test_metrics_rnn[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[:24, 0].mean()))
# test accuracy: 0.63273

loss_pred_plots(history=history_rnn, skip_epoch=0, model=model_rnn,
                test=test_gen, test_target=test_data[:, 0], pred_periods=48)

predictions = model_rnn.predict_generator(test_gen, steps=1)
predictions.shape

# (8064, 1) for steps 48
# (4032, 1) for steps 24
# (168, 1) for steps 1

plt.plot(predictions)
plt.show()

plt.plot(test_data[:4032, 0])
plt.show()

import pickle

with open(f'models/rnn_20.pickle', 'wb') as pfile:
    pickle.dump(model_rnn, pfile)
with open(f'models/rnn_20_hist.pickle', 'wb') as pfile:
    pickle.dump(history_rnn, pfile)