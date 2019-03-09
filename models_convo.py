import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import datetime as dt
import statsmodels.formula.api as sm


import seaborn as sns
sns.set()

with open('test_big.csv', 'r') as f:
    df = pd.read_csv(f, index_col=0)

df.columns = pd.to_datetime(df.columns)
data_start_date = df.columns[0]
data_end_date = df.columns[-1]
print('Data ranges from %s to %s' % (data_start_date, data_end_date))

# train/val split
pred_steps = 168
pred_length = dt.timedelta(hours=pred_steps)

val_pred_start = data_end_date - pred_length + dt.timedelta(hours=1)
val_pred_end = data_end_date

train_pred_start = val_pred_start - pred_length
train_pred_end = val_pred_start - dt.timedelta(hours=1)

enc_length = train_pred_start - data_start_date

train_enc_start = data_start_date
train_enc_end = train_enc_start + enc_length - dt.timedelta(hours=1)

val_enc_start = train_enc_start + pred_length
val_enc_end = val_enc_start + enc_length - dt.timedelta(hours=1)

print('Train encoding:', train_enc_start, '-', train_enc_end)
print('Train prediction:', train_pred_start, '-', train_pred_end)
print('Val encoding:', val_enc_start, '-', val_enc_end)
print('Val prediction:', val_pred_start, '-', val_pred_end)

# format inputs
date_to_index = pd.Series(index=pd.Index([c for c in df.columns]),
                          data=[i for i in range(len(df.columns))])


def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]


def transform_series_encode(series_array):

    series_mean = series_array.mean(axis=1).reshape(-1, 1)
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array, series_mean


def transform_series_decode(series_array, encode_series_mean):

    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate
from keras.optimizers import Adam

# convolutional layer parameters
n_filters = 32
filter_width = 2
dilation_rates = [2**i for i in range(8)]

# define an input history series and pass it through a stack of dilated causal convolutions
history_seq = Input(shape=(None, 1))
x = history_seq

for dilation_rate in dilation_rates:
    x = Conv1D(filters=n_filters,
               kernel_size=filter_width,
               padding='causal',
               dilation_rate=dilation_rate)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(.2)(x)
x = Dense(1)(x)

# extract the last 168 time steps as the training target
def slice(x, seq_length):
    return x[:,-seq_length:,:]

pred_seq_train = Lambda(slice, arguments={'seq_length':168})(x)

model = Model(history_seq, pred_seq_train)

model.summary()

# train the model
batch_size = 32
epochs = 10

# sample of series from train_enc_start to train_enc_end
encoder_input_data = get_time_block_series(df.values, date_to_index,
                                           train_enc_start, train_enc_end)
encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

# sample of series from train_pred_start to train_pred_end
decoder_target_data = get_time_block_series(df.values, date_to_index,
                                            train_pred_start, train_pred_end)
decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

# ...teacher forcing, where during training, the true series values (lagged by one time step)
# are fed as inputs to the decoder. Intuitively, we are trying to teach the NN how to condition on
# previous time steps to predict the next. At prediction time, the true values in this process
# will be replaced by predicted values for each previous time step.

# we append a lagged history of the target series to the input data,
# so that we can train with teacher forcing
lagged_target_history = decoder_target_data[:, :-1, :1]
encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

model.compile(Adam(), loss='mean_absolute_error')
history = model.fit(encoder_input_data, decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2)