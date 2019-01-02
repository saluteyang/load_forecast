import csv
import datetime as dt
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import holidays
from keras import models
from keras import layers
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
weather_hr = weather.groupby([lambda x: x.date, lambda x: x.hour])['Drybulb', 'Humidity'].mean()
# plot using index needs to turn index into a column first
# note that multi-index will be turned into columns by level
weather_hr['Hour_End'] = pd.to_datetime(weather_hr.index.map(lambda x: '-'.join((str(x[0]), str(x[1])))), format='%Y-%m-%d-%H')
weather_hr.plot(x='Hour_End', y='Drybulb')
plt.show()

# weather['Date'] = weather['Hour_End'].dt.date
# weather['Hour'] = weather['Hour_End'].dt.hour
# weather_grouped = weather.groupby(['Date', 'Hour'])
# weather_hr = weather_grouped.agg({'Drybulb': np.average, 'Humidity': np.average})
# weather_hr = weather_hr.reset_index()
# weather_hr.plot(x='Date', y='Drybulb')
# plt.show()

# import actual load including profiles from two business and residential profiles

aggregate_load = pd.DataFrame()

# run the below code if need to reprocess from source spreadsheets
# or skip and import processed data
##########
# for f in glob.glob('ERCOT_load_profiles/*native*.csv'):
#     print(f)
#     df = pd.read_csv(f)
#     # print(df.columns)
#     aggregate_load = aggregate_load.append(df, ignore_index=True)
#
# aggregate_load['Hour_End'] = pd.to_datetime(aggregate_load['Hour_End'])
# aggregate_load.iloc[:, 1:10].astype('float')
# aggregate_load.to_csv('test.csv', index=False)
##########
# above will fail initially as there is 24 in the hour string in the 2017 file
# there is also a thousands separator inside numbers in that file that causes parsing problems

# aggregate_load['Date'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[0]
# aggregate_load['Hour'] = aggregate_load['Hour_End'].str.split(' ', expand=True)[1]
# aggregate_load['Hour'] = aggregate_load['Hour'].str.split(':', expand=True)[0]
# set(aggregate_load['Hour'])
# aggregate_load[aggregate_load['Hour'] == '24']

# start here to import processed data, if using other profiles, run section in profile_proc.py ###############
with open('test.csv', 'r') as f:
    aggregate_load = pd.read_csv(f, index_col=0)
aggregate_load['COAST'] = aggregate_load['COAST']/1000

# plot time series with matplotlib
aggregate_load.plot(x='Hour_End', y='ERCOT')
plt.show()

# join weather and load information
aggregate_load = aggregate_load.set_index('Hour_End')
aggregate_load.index = pd.to_datetime(aggregate_load.index)

# section end, joining weather and load data begins below ##############################################

weather_hr = weather_hr.set_index('Hour_End')
joined = aggregate_load.join(weather_hr, how='inner')

joined[['COAST', 'Drybulb', 'Humidity']].corr()

lm = sm.ols(formula='COAST ~ Drybulb + Humidity', data=joined).fit()
lm.params
lm.summary()

# the relationship is not linear
joined.plot(x='Drybulb', y='COAST', kind='scatter')
plt.show()

# len(joined.index.unique())  # duplicate index due to additional hour in Nov due to DST

joined = joined.groupby(joined.index).first()
joined_keep = joined[['COAST', 'Drybulb', 'Humidity']].copy()
# run up to the line above for recurrent network test #################################################

joined_keep['Hour_Num'] = joined_keep.index.hour
joined_keep['Day_Num'] = joined_keep.index.weekday  # Monday is 0
joined_keep['Wknd_Flag'] = (joined_keep.index.weekday > 4) * 1

joined_keep['Date'] = joined_keep.index.date

# add holidays flag
us_holidays = holidays.UnitedStates()
# for hld in holidays.UnitedStates(years=[2010, 2011]).items():
#     print(hld)
joined_keep['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined_keep['Date']]
# to explicitly use the lambda function for the same effect as above
# joined_keep['Holiday_Flag'] = [(lambda x: (x in us_holidays) * 1)(x) for x in joined_keep['Date']]
joined_keep['Off_Flag'] = joined_keep[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)


joined_keep = joined_keep.merge(joined_keep.groupby(joined_keep.index.date)['COAST'].mean().to_frame(),
                                left_on='Date', right_index=True,
                                suffixes=['_Hourly', '_DailyAve'])  # syntax different from join method
joined_keep = joined_keep.merge(joined_keep['COAST_Hourly'].shift(1, 'd').to_frame(),
                                left_index=True, right_index=True, how='outer',
                                suffixes=['', '_Pre_Day'])
joined_keep = joined_keep.merge(joined_keep['COAST_Hourly'].shift(7, 'd').to_frame(),
                                left_index=True, right_index=True, how='outer',
                                suffixes=['', '_Pre_Wk_Day'])
joined_keep = joined_keep.merge(joined_keep['COAST_DailyAve'].shift(1, 'd').to_frame(),
                                left_index=True, right_index=True, how='outer',
                                suffixes=['', '_Pre_Day'])

# examine rows that contain NaN
# joined_keep.dropna().shape
# joined_keep[joined_keep.isnull().any(axis=1)]

joined_keep = joined_keep.dropna()
joined_keep = joined_keep.drop(columns=['Date', 'COAST_DailyAve', 'Wknd_Flag', 'Holiday_Flag'])
joined_keep_float = joined_keep.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity'])
joined_keep_int = joined_keep[['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity']]
joined_keep_int = joined_keep_int.astype('int')
joined_clean = joined_keep_float.merge(joined_keep_int, left_index=True, right_index=True)

joined_clean_train = joined_clean[joined_clean.index.year!=2017]
train_data = joined_clean_train.drop(columns=['COAST_Hourly'])
train_target = joined_clean_train['COAST_Hourly']
joined_clean_test = joined_clean[joined_clean.index.year==2017]
test_data = joined_clean_test.drop(columns=['COAST_Hourly'])
test_target = joined_clean_test['COAST_Hourly']

# model1: using regression model

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model()
partial_train_data = train_data[10000:]
val_data = train_data[:10000]
partial_train_target = train_target[10000:]
val_target = train_target[:10000]
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

# results: mae on training 0.4993, on validation 0.6769, on test 0.5430
# results: mae on training 0.2608, on validation 0.3438, on test 0.3932 (busmed)
# results: mae on training 0.0599, on validation 0.0494, on test 0.0552 (reslo)

# start here for all the RNN models and then run the specific code under each #########################
# generators for training, validation and test sets

def generator(data, lookback, delay, min_index,
              max_index, batch_size=168):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if i + batch_size >= max_index:
            i = min_index + lookback
        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = np.zeros((len(rows),
                            lookback,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j])
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets

lookback = 336  # 2 weeks
delay = 24  # 1 day ahead
batch_size = 168  # 7 days in each batch

# steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
# validation_steps = TotalValidationSamples / ValidationBatchSize
# val_steps = (60000 - 50001 - lookback)  # how many steps to use up the entire validation set ???
# test_steps = (len(joined_keep) - 60001 - lookback)

# model2: using recurrent network

joined_keep = joined_keep.dropna()
joined_keep = joined_keep.values  # turn into numpy array

train_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=50000,
                      batch_size=batch_size)

val_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=50001,
                      max_index=60000,
                      batch_size=batch_size)

test_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=60001,
                      max_index=None,
                      batch_size=batch_size)

def build_model2():
    model = models.Sequential()
    model.add(layers.GRU(32, return_sequences=True, input_shape=(None, joined_keep.shape[-1])))
    model.add(layers.GRU(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model2 = build_model2()
history = model2.fit_generator(train_gen,
                              steps_per_epoch=290,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=55)

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

predictions = model2.predict_generator(test_gen, steps=55)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), joined_keep[:1000, 0], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model2.evaluate_generator(test_gen, steps=55)

# experiments to improve accuracy:
# have attempted different number of hidden units, including second GRU layer,
# model3: using LSTM instead of GRU and increasing epochs

joined_keep = joined_keep.dropna()
joined_keep = joined_keep.values  # turn into numpy array

train_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=50000,
                      batch_size=batch_size)

val_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=50001,
                      max_index=60000,
                      batch_size=batch_size)

test_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=60001,
                      max_index=None,
                      batch_size=batch_size)

def build_model3():
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=(None, joined_keep.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model3 = build_model3()
history = model3.fit_generator(train_gen,
                              steps_per_epoch=290,
                              epochs=15,
                              validation_data=val_gen,
                              validation_steps=55)

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

predictions = model3.predict_generator(test_gen, steps=55)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), joined_keep[:1000, 0], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model3.evaluate_generator(test_gen, steps=55)

# model4: include the calendar dummies in the input data

joined_keep['Hour_Num'] = joined_keep.index.hour
joined_keep['Day_Num'] = joined_keep.index.weekday  # Monday is 0
joined_keep['Wknd_Flag'] = (joined_keep.index.weekday > 4) * 1
joined_keep['Date'] = joined_keep.index.date

us_holidays = holidays.UnitedStates()

joined_keep['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined_keep['Date']]
joined_keep['Off_Flag'] = joined_keep[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)
joined_keep = joined_keep.dropna()
joined_keep = joined_keep.drop(columns=['Date', 'Wknd_Flag', 'Holiday_Flag'])
joined_keep_float = joined_keep.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity'])
joined_keep_int = joined_keep[['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity']]
joined_keep_int = joined_keep_int.astype('int')
joined_keep = joined_keep_float.merge(joined_keep_int, left_index=True, right_index=True)
joined_keep = joined_keep.values

train_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=50000,
                      batch_size=batch_size)

val_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=50001,
                      max_index=60000,
                      batch_size=batch_size)

test_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=60001,
                      max_index=None,
                      batch_size=batch_size)

def build_model4():
    model = models.Sequential()
    model.add(layers.GRU(32, return_sequences=True, input_shape=(None, joined_keep.shape[-1])))
    model.add(layers.GRU(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model4 = build_model4()
history = model4.fit_generator(train_gen,
                              steps_per_epoch=290,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=55)

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

predictions = model4.predict_generator(test_gen, steps=55)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), joined_keep[:1000, 0], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model4.evaluate_generator(test_gen, steps=55)

# model5: add a dense layer before the GRU layer, keeping dummies

joined_keep['Hour_Num'] = joined_keep.index.hour
joined_keep['Day_Num'] = joined_keep.index.weekday  # Monday is 0
joined_keep['Wknd_Flag'] = (joined_keep.index.weekday > 4) * 1
joined_keep['Date'] = joined_keep.index.date

us_holidays = holidays.UnitedStates()

joined_keep['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined_keep['Date']]
joined_keep['Off_Flag'] = joined_keep[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)
joined_keep = joined_keep.dropna()
joined_keep = joined_keep.drop(columns=['Date', 'Wknd_Flag', 'Holiday_Flag'])
joined_keep_float = joined_keep.drop(columns=['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity'])
joined_keep_int = joined_keep[['Hour_Num', 'Day_Num', 'Off_Flag', 'Drybulb', 'Humidity']]
joined_keep_int = joined_keep_int.astype('int')
joined_keep = joined_keep_float.merge(joined_keep_int, left_index=True, right_index=True)
joined_keep = joined_keep.values

train_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=50000,
                      batch_size=batch_size)

val_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=50001,
                      max_index=60000,
                      batch_size=batch_size)

test_gen = generator(joined_keep,
                      lookback=lookback,
                      delay=delay,
                      min_index=60001,
                      max_index=None,
                      batch_size=batch_size)

def build_model5():
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu',
                           input_shape=(None, joined_keep.shape[-1])))
    model.add(layers.GRU(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model5 = build_model5()
history = model5.fit_generator(train_gen,
                              steps_per_epoch=290,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=55)

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

predictions = model5.predict_generator(test_gen, steps=55)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), joined_keep[:1000, 0], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model5.evaluate_generator(test_gen, steps=55)

# results: mae on training 0.7444, on validation 1.1637, on test 1.3112 (20 epochs)
# results: mae on training 0.6535, on validation 1.5005, on test 1.6972 (40 epochs)
