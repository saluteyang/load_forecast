import csv
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import holidays
from keras import models
from keras import layers
import os
import statsmodels.tsa.api as smt
import seaborn as sns
import matplotlib.gridspec as gs
from sklearn.tree import DecisionTreeRegressor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# section A: weather information

filename = 'Houston_tx_hobby_2010-2017.csv'

# choose only REPORTTPYE (misspelt) FM-15
# exclude time periods when hourly drybulb or humidity data are missing
# missing data can be blanks or wildcard character *

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    # print header with position
    for index, column_header in enumerate(header_row):
        print(index, column_header)

    dates, drybulb, humidity = [], [], []
    for row in reader:
        if row[6] == 'FM-15':
            try:
                date_obs = dt.datetime.strptime(row[5], "%Y-%m-%d %H:%M")
                drybulb_temp = int(row[10])
                relative_humidity = int(row[16])

            except ValueError:
                print(date_obs, 'missing data')

            else:
                dates.append(date_obs)
                drybulb.append(drybulb_temp)
                humidity.append(relative_humidity)

weather = pd.DataFrame({'Hour_End': dates,
                        'Drybulb': drybulb,
                        'Humidity': humidity})
weather = weather.set_index('Hour_End')

# average out sub-hour weather info
weather = weather.groupby([lambda x: x.date, lambda x: x.hour])['Drybulb', 'Humidity'].mean()
weather.index = pd.to_datetime(weather.index.map(lambda x: '-'.join((str(x[0]), str(x[1])))), format='%Y-%m-%d-%H')

# section B: joining weather and load information and time series plots
# import processed data; if using other profiles, run section in profile_proc.py and continue below ###############
aggregate_load = pd.DataFrame()
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000  # source data units are in MW, here converted to GW
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)

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

partial_train_data = train_data[10000:]
val_data = train_data[:10000]
partial_train_target = train_target[10000:]
val_target = train_target[:10000]

# optional plots for time series EDA
# visualizing load
def load_tsplots():
    subsample = aggregate_load['COAST'][:336]

    plt.figure()
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    subsample.plot(ax=ts_ax)
    ts_ax.set_title('')
    ts_ax.set_xlabel('')
    # ts_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # did not work; import matplotlib.dates as mdates
    # for tick in ts_ax.get_xticklabels():
    #     tick.set_rotation(90)
    subsample.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(subsample, lags=48, ax=acf_ax)
    smt.graphics.plot_pacf(subsample, lags=48, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    plt.show()

# visualizing load/temperature relationship
def load_temp_distplot():
    plt.figure()
    gspec = gs.GridSpec(3, 3)
    top_hist = plt.subplot(gspec[0, 1:]) # index position starts with 0
    side_hist = plt.subplot(gspec[1:, 0])
    lower_right = plt.subplot(gspec[1:, 1:])

    top_hist.hist(joined_keep['Drybulb'], normed=True)
    side_hist.hist(joined_keep['COAST_Hourly'], bins=50, orientation='horizontal', normed=True)
    side_hist.invert_xaxis()
    lower_right.scatter(joined_keep.Drybulb, joined_keep.COAST_Hourly)
    lower_right.set_xlabel('temperature (F)')
    lower_right.set_ylabel('load (GW)')
    plt.show()

# section C: different modelling approaches
# decision tree model (baseline?)  ###########################################################################

model0 = DecisionTreeRegressor(min_samples_leaf=20).fit(train_data, train_target)

print('training accuracy: {:.3f}'.format(model0.score(train_data, train_target)))
print('test accuracy: {:.3f}'.format(model0.score(test_data, test_target)))

predictions = model0.predict(test_data)
plt.plot(test_target.index[:1000], test_target[:1000], 'b', label='test actual', alpha=0.2)
plt.plot(test_target.index[:1000], predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

# training accuracy: 0.960 (here the score method returns R-square)
# test accuracy: 0.903

# using accuracy definition as below
mae_pct = 1 - np.mean(abs(predictions - test_target.values.reshape(len(test_target.values), 1)))/np.mean(test_target.values)
# test accuracy: 0.95555

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

loss = history.history['loss']
val_loss = history.history['val_loss']
history_dict = history.history
epochs = range(1, len(history_dict['loss']) + 1)

# leave out the first point
plt.plot(epochs[1:], loss[1:], 'bo', label='training loss')  # blue dot
plt.plot(epochs[1:], val_loss[1:], 'b', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(test_data)
plt.plot(test_target.index[:1000], test_target[:1000], 'b', label='test actual', alpha=0.2)
plt.plot(test_target.index[:1000], predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model.evaluate(test_data, test_target)
test_mae = model.evaluate(test_data, test_target)[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_target.mean()))

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

# redimension datasets for RNN
def rnn_redim(data, lookback=336):
    rows = np.arange(lookback, data.shape[0])
    samples = np.zeros((len(rows), lookback, data.shape[1]))
    for j, row in enumerate(rows):
        indices = range(rows[j] - lookback, rows[j])
        samples[j] = data[indices]
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

loss = history.history['loss']
val_loss = history.history['val_loss']
history_dict = history.history
epochs = range(1, len(history_dict['loss']) + 1)

plt.plot(epochs[2:], loss[2:], 'bo', label='training loss')  # blue dot
plt.plot(epochs[2:], val_loss[2:], 'b', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model_corrected.predict(test_data_3d)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), test_target[:1000], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

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
# test accuracy: 0.94259 (for 20 epochs-rerun)
# test accuracy: 0.91674 (for 20 epochs-no lagged)
# test accuracy: 0.94192 (for 20 epochs-no dummies)
# test accuracy: 0.97236 (for 20 epochs-no weather & dummies)
# test accuracy: 0.97768 (for 20 epochs-no weather & dummies-rerun)
