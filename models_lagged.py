import pandas as pd
import numpy as np
import holidays
from keras import models, layers
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import pickle
from helpers import *

sns.set()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# section A: weather information
with open('houston_weather.csv', 'r') as f:
    weather = pd.read_csv(f, index_col=0)

# section B: joining weather and load information and time series plots
# import processed data; if using other profiles, run section in profile_proc.py and continue below ###############
# aggregate_load = pd.DataFrame()
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000  # source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)

# if running without weather data
# duplicate index due to additional hour in Nov due to DST
joined = aggregate_load.groupby(aggregate_load.index).first()
joined = joined.dropna().copy()
joined = joined['COAST']

# if running with weather data ##########################################
joined = aggregate_load.join(weather, how='inner')  # joining on index
joined = joined.groupby(joined.index).first()  # duplicate index due to additional hour in Nov due to DST
joined_keep = joined[['COAST', 'Drybulb', 'Humidity']].dropna().copy()

# create indicator variables
joined_keep['Hour_Num'] = joined_keep.index.hour
joined_keep['Day_Num'] = joined_keep.index.weekday  # Monday is 0
joined_keep['Wknd_Flag'] = (joined_keep.index.weekday > 4) * 1
joined_keep['Date'] = joined_keep.index.date

# add holidays flag
us_holidays = holidays.UnitedStates()  # this creates a dictionary
joined_keep['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined_keep['Date']]
# to explicitly use the lambda function for the same effect as above
# joined_keep['Holiday_Flag'] = [(lambda x: (x in us_holidays) * 1)(x) for x in joined_keep['Date']]
joined_keep['Off_Flag'] = joined_keep[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)

# create lagged variables
joined_keep = joined_keep.merge(joined_keep.groupby(joined_keep.index.date)['COAST'].mean().to_frame(),
                                left_on='Date', right_index=True,
                                suffixes=['_Hourly', '_DailyAve'])  # syntax different from join method
joined_keep['COAST_Hourly_Pre_Day'] = joined_keep['COAST_Hourly'].shift(1, 'd').to_frame()
joined_keep['COAST_Hourly_Pre_Wk_Day'] = joined_keep['COAST_Hourly'].shift(7, 'd').to_frame()
joined_keep['COAST_DailyAve_Pre_Day'] = joined_keep['COAST_DailyAve'].shift(1, 'd').to_frame()

# joined_keep[joined_keep.isnull().any(axis=1)]
joined_keep = joined_keep.dropna()
joined_keep = joined_keep.drop(columns=['Date', 'COAST_DailyAve', 'Wknd_Flag', 'Holiday_Flag'])
int_cols = ['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity']
joined_keep[int_cols] = joined_keep[int_cols].applymap(lambda x: int(x))

# create training and testing data sets
train_data = joined_keep[joined_keep.index.year != 2017].drop(columns=['COAST_Hourly'])
train_target = joined_keep[joined_keep.index.year != 2017]['COAST_Hourly']
test_data = joined_keep[joined_keep.index.year == 2017].drop(columns=['COAST_Hourly'])
test_target = joined_keep[joined_keep.index.year == 2017]['COAST_Hourly']
###############################################################################
# two corrections (train, val time order; standardization)
# create lagged variables
joined = joined.to_frame().merge(joined.groupby(joined.index.date).mean().to_frame(),
                                left_index=True, right_index=True,
                                suffixes=['_Hourly', '_DailyAve'], how='outer')  # syntax different from join method
joined = joined.fillna(method='ffill')
joined['COAST_Hourly_Pre_Day'] = joined['COAST_Hourly'].shift(1, 'd').to_frame()
joined['COAST_Hourly_Pre_Wk_Day'] = joined['COAST_Hourly'].shift(7, 'd').to_frame()
joined['COAST_DailyAve_Pre_Day'] = joined['COAST_DailyAve'].shift(1, 'd').to_frame()
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

# section C: different modelling approaches
# decision tree model (baseline?)  ###########################################################################

model0 = DecisionTreeRegressor(min_samples_leaf=20).fit(train_data, train_target)

y_pred = model0.predict(test_data)

print('training accuracy: {:.3f}'.format(model0.score(train_data, train_target)))
print('test accuracy: {:.3f}'.format(model0.score(test_data, test_target)))

pred_plot(model=model0, test=test_data, test_target=test_target, pred_periods=168)

# training accuracy: 0.960 (here the score method returns R-square)
# test accuracy: 0.903

from sklearn.ensemble import GradientBoostingRegressor

alpha = 0.95
model1 = GradientBoostingRegressor(min_samples_leaf=20, alpha=alpha, loss='quantile',
                                   n_estimators=100)
model1.fit(train_data, train_target)

# make predictions
y_upper = model1.predict(test_data)

model1.set_params(alpha=1.0 - alpha)
model1.fit(train_data, train_target)

y_lower = model1.predict(test_data)

model1.set_params(loss='ls')
model1.fit(train_data, train_target)

y_pred = model1.predict(test_data)

print('training accuracy: {:.3f}'.format(model1.score(train_data, train_target)))
print('test accuracy: {:.3f}'.format(model1.score(test_data, test_target)))

# plot 1 week actual and prediction interval
periods = 168
plt.plot(y_upper[:periods])
plt.plot(y_lower[:periods])
plt.plot(y_pred[:periods])
plt.plot(test_target.values[:periods], linestyle='dashed', color='black')
plt.show()


# training accuracy: 0.952
# test accuracy: 0.921

# using accuracy definition as below
# mae_pct = 1 - np.mean(abs(y_pred - test_target.values.reshape(len(test_target.values), 1)))/np.mean(test_target.values)
mae_pct = 1 - np.mean(np.absolute(y_pred - test_target.values))/np.mean(test_target.values)
# test accuracy: 0.94985 (decision tree)
# test accuracy: 0.95358 (gradient-boosted trees)

# regression model  #########################################################################################

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # in Keras, accuracy is a classification metric
    return model

model = build_model()
history = model.fit(partial_train_data, partial_train_target,
                    epochs=60, batch_size=168,
                    validation_data=(val_data, val_target))

loss_plot(history=history, skip_epoch=1)
pred_plot(model=model, test=test_data, test_target=test_target, pred_periods=168)

model.evaluate(test_data, test_target)
test_mae = model.evaluate(test_data, test_target)[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_target.mean()))
# test accuracy: 0.96325 (final)

# with open(f'models/dense_60_lag.pickle', 'wb') as pfile:
#     pickle.dump(model, pfile)
# with open(f'models/dense_60_lag_hist.pickle', 'wb') as pfile:
#     pickle.dump(history, pfile)

# test accuracy: 0.95497

# results: mae on training 0.4993, on validation 0.6769, on test 0.5430
# results: mae on training 0.2608, on validation 0.3438, on test 0.3932 (busmed)
# results: mae on training 0.0599, on validation 0.0494, on test 0.0552 (reslo)

# start here for all the RNN models ####################################################################

# variation 1: remove lagged features (optional)
# train_data = train_data[['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity']]
# test_data = test_data[['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity']]

# variation 2: remove dummy features (optional)
# train_data = train_data.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag'])
# test_data = test_data.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag'])

# variation 3: remove weather and dummy features (optional)
# train_data = train_data.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity'])
# test_data = test_data.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity'])

partial_train_data = partial_train_data.values
partial_train_target = partial_train_target.values[:-336]
test_data = test_data.values
test_target = test_target.values[:-336]
val_data = val_data.values
val_target = val_target.values[:-336]

# use this part with final version
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

loss_plot(history=history, skip_epoch=1)
pred_plot(model=model_corrected, test=test_data_3d, test_target=test_target, pred_periods=168)

# model_corrected.evaluate(test_data_3d, test_target)

# results: mae on training 0.4560, on validation 0.3930, on test 0.4813 (20 epochs)
# results: mae on training 0.5451, on validation 0.5451, on test 0.6878 (20 epochs-rerun)
# results: mae on training 0.4707, on validation 0.6406, on test 0.8391 (30 epochs)
# results: mae on training 0.3860, on validation 0.4410, on test 0.6136 (40 epochs)
# results: mae on training 0.6720, on validation 0.7920, on test 0.9976 (20 epochs, no lagged)
# results: mae on training 0.5543, on validation 0.5389, on test 0.6959 (20 epochs, no dummies)
# results: mae on training 0.2796, on validation 0.3351, on test 0.3312 (20 epochs, no weather & dummies)
# results: mae on training 0.2871, on validation 0.2344, on test 0.2674 (20 epochs, no weather & dummies-rerun)

test_mae = model_corrected.evaluate(test_data_3d, test_target)[1]
print('test accuracy: {:.5f}'.format(1-test_mae/test_target.mean()))

# test accuracy: 0.96953 (final)

# with open(f'models/rnn_20_lag.pickle', 'wb') as pfile:
#     pickle.dump(model, pfile)
# with open(f'models/rnn_20_lag_hist.pickle', 'wb') as pfile:
#     pickle.dump(history, pfile)

with open(f'models/rnn_20_lag_nw_wd.pickle', 'wb') as pfile:
    pickle.dump(model_corrected, pfile)
with open(f'models/rnn_20_lag_nw_wd_hist.pickle', 'wb') as pfile:
    pickle.dump(history, pfile)


# test accuracy: 0.94259 (for 20 epochs-rerun)
# test accuracy: 0.91674 (for 20 epochs-no lagged)
# test accuracy: 0.94192 (for 20 epochs-no dummies)
# test accuracy: 0.97236 (for 20 epochs-no weather & dummies)
# test accuracy: 0.97040 (for 20 epochs-no weather, with dummies)
# test accuracy: 0.97768 (for 20 epochs-no weather & dummies-rerun)
