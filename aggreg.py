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
# the choices are:
# 1) include/exclude indicator variables (this does not hurt but Dense model predictions can be jagged)
# 2) include/exclude weather variables (this helps minimally with RNN presumably because weather is not location specific
# enough and or RNN has already learned enough pattern form the past that weather provide small incremental information
# 3) include/exclude lagged variables (this is only for the daily update frequency models)
def proc_load(file_name):
    with open(file_name, 'r') as f:
        aggregate_load = pd.read_csv(f, index_col=0)

    # source data units are in MW, here converted to GW
    aggregate_load = aggregate_load.set_index('Hour_End')
    aggregate_load = aggregate_load.apply(lambda x: x/1000)
    aggregate_load.index = pd.to_datetime(aggregate_load.index)

    # duplicate index due to additional hour in Nov due to DST
    joined = aggregate_load.groupby(aggregate_load.index).first()
    joined = joined.dropna().copy()
    return joined


def indicator_lag(df, region='COAST', dummy=True, lag=True):
    # specify the region to forecast
    region = region
    cols = [x for x in df.columns if x!=region]  # get region columns to be dropped later
    if dummy:
        # create indicator variables
        df['Hour_Num'] = df.index.hour
        df['Day_Num'] = df.index.weekday  # Monday is 0
        df['Wknd_Flag'] = (df.index.weekday > 4) * 1
        df['Date'] = df.index.date
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['Day_Year'] = df.index.dayofyear
        df['Week_Year'] = df.index.weekofyear

        # add holidays flag
        us_holidays = holidays.UnitedStates()  # this creates a dictionary
        df['Holiday_Flag'] = [(x in us_holidays) * 1 for x in df['Date']]
        df['Off_Flag'] = df[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)
        if lag is False:
            df = df.drop(columns=cols + ['Date', 'Wknd_Flag', 'Holiday_Flag'])
        else:
            df = df.drop(columns=cols + ['Wknd_Flag', 'Holiday_Flag'])
    if lag:
        # create lagged variables
        df = df.merge(df.groupby(df.index.date)[region].mean().to_frame(),
                      left_on='Date', right_index=True,
                      suffixes=['_Hourly', '_DailyAve'])  # syntax different from join method
        df[region + '_Hourly_Pre_Day'] = df[region + '_Hourly'].shift(1, 'd').to_frame()
        df[region + '_Hourly_Pre_Wk_Day'] = df[region + '_Hourly'].shift(7, 'd').to_frame()
        df[region + '_DailyAve_Pre_Day'] = df[region + '_DailyAve'].shift(1, 'd').to_frame()
        df = df.drop(columns=['Date', 'COAST_DailyAve'])

    df = df.dropna()
    return df


def train_test(df, region='COAST', for_gen=False, test_year=2017, val_split=True, val_num=10000):
    if for_gen:
        # create training and testing data sets, generator will separate out the features and target
        train_data = df[df.index.year != test_year]
        test_data = df[df.index.year == test_year]

    else:
        train_data = df[df.index.year != test_year].drop(columns=region+'_Hourly')
        train_target = df[df.index.year != test_year][region+'_Hourly']
        test_data = df[df.index.year == test_year].drop(columns=region+'_Hourly')
        test_target = df[df.index.year == test_year]['COAST_Hourly']

    # if one_col:
    #     # standardization for no dummy version
    #     scaler = preprocessing.MinMaxScaler()
    #     train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))  # -1 asks np to figure it out
    #     test_data = scaler.transform(test_data.values.reshape(-1, 1))
    # else:
    #     # added section for standardization
    #     scaler = preprocessing.MinMaxScaler()
    #     train_data = scaler.fit_transform(train_data)
    #     test_data = scaler.transform(test_data)

    scaler = preprocessing.MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    if val_split:
        # all but the last val_num used for training, val is the rest
        partial_train_data = train_data[:-val_num]
        val_data = train_data[-val_num:]
        partial_train_target = train_target[:-val_num]
        val_target = train_target[-val_num:]

        partial_train_target = partial_train_target.values[336:]
        test_target = test_target.values[336:]
        val_target = val_target.values[336:]

    if for_gen:
        return train_data, test_data, [], [], [], [], [], []
    else:
        # for DecisionTreeRegressor and GradientBoostingRegressor train_data/target and test_data/target are used
        # for RNN/Dense models, partial_train_data/target, val_data/target and test_data/target are used
        return train_data, test_data, train_target, test_target, partial_train_data, val_data, partial_train_target, val_target


# create generators using udf in helpers file
def create_gen(train_data, test_data, lookback=1440, delay=24):
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
    return train_gen, val_gen, test_gen


# redimension datasets for RNN (only applies for RNN model with non-generator inputs
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


def with_rnn(partial_train_data, val_data, test_data):
    partial_train_data_3d = rnn_redim(partial_train_data)
    val_data_3d = rnn_redim(val_data)
    test_data_3d = rnn_redim(test_data)
    return partial_train_data_3d, val_data_3d, test_data_3d


# defining model architecture for RNN model
def build_model_rnn(train_data):
    model = models.Sequential()
    model.add(layers.GRU(64, input_shape=(None, train_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# defining model architecture for Dense model
def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(1440, train_data.shape[-1])))
    # model.add(layers.Flatten(input_shape=(lookback, 1)))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(40, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



def choose_model(model, use_gen=False, out=True,
                 train_data=None, test_data=None, partial_train_data=None, partial_train_target=None,
                 val_data=None, val_target=None):

    # redimension data for non-generator version of RNN
    if model == 'RNN':
        partial_train_data_3d, val_data_3d, test_data_3d = with_rnn(partial_train_data, val_data, test_data)

    # create generators if used
    if use_gen:
        train_gen, val_gen, test_gen = create_gen(train_data, test_data)

    if model == 'Dense'and use_gen:
        model = build_model(train_data)
        history = model.fit_generator(train_gen,
                                      epochs=20,
                                      steps_per_epoch=365,
                                      validation_data=val_gen,
                                      validation_steps=50,
                                      callbacks=[
                                          callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
                                      ])
    elif model == 'Dense' and use_gen is False:
        model = build_model(train_data)
        history = model.fit(partial_train_data, partial_train_target,
                            epochs=60, batch_size=168,
                            validation_data=(val_data, val_target))

    elif model == 'RNN' and use_gen is False:
        model_rnn = build_model_rnn(train_data)
        history_rnn = model_rnn.fit(partial_train_data_3d,
                                    partial_train_target,
                                    epochs=20, batch_size=168,
                                    validation_data=(val_data_3d, val_target))
    elif model == 'RNN' and use_gen:
        model_rnn = build_model_rnn(train_data)
        history_rnn = model_rnn.fit_generator(train_gen,
                                              steps_per_epoch=365,
                                              epochs=20,
                                              validation_data=val_gen,
                                              validation_steps=50
                                              )
    else:
        print('This configuration of model has not been set up.')

    if out and model == 'RNN':
        with open(f'temp_out/rnn_temp.pickle', 'wb') as pfile:
            pickle.dump(model_rnn, pfile)
        with open(f'temp_out/rnn_temp_hist.pickle', 'wb') as pfile:
            pickle.dump(history_rnn, pfile)
    elif out and model == 'Dense':
        with open(f'temp_out/dense_temp.pickle', 'wb') as pfile:
            pickle.dump(model, pfile)
        with open(f'temp_out/dense_temp_hist.pickle', 'wb') as pfile:
            pickle.dump(history, pfile)


def main():
    print('processing input')
    load = proc_load('test.csv')
    print('creating dummies and/or lagging variables ')
    load = indicator_lag(load, region='COAST', dummy=True, lag=True)
    print('create train/test datasets and separate target when necessary')
    train_data, test_data, train_target, test_target, partial_train_data, val_data, partial_train_target, val_target = \
        train_test(load, region='COAST', for_gen=False, test_year=2017, val_split=True, val_num=10000)
    print('fitting defined models and writing output if desired')
    choose_model(model='Dense', use_gen=True, out=True, train_data=train_data, test_data=test_data)
    print('modelling process complete')


if __name__ == '__main__':
    main()
