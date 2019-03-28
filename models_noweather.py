import pandas as pd
import numpy as np
import holidays
from keras import models, layers, callbacks
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# specify the region to forecast
region = 'COAST'
cols = [x for x in joined.columns if x!=region]

# create indicator variables
# joined['Hour_Num'] = joined.index.hour
# joined['Day_Num'] = joined.index.weekday  # Monday is 0
# joined['Month'] = joined.index.month
# joined['Day'] = joined.index.day
# joined['Week_Year'] = joined.index.weekofyear
joined['Date'] = joined.index.date
joined['Wknd_Flag'] = (joined.index.weekday > 4) * 1
joined['Day_Year'] = joined.index.dayofyear

# add holidays flag
us_holidays = holidays.UnitedStates()  # this creates a dictionary
joined['Holiday_Flag'] = [(x in us_holidays) * 1 for x in joined['Date']]
joined['Off_Flag'] = joined[['Wknd_Flag', 'Holiday_Flag']].max(axis=1)

joined = joined.drop(columns=cols + ['Date', 'Wknd_Flag', 'Holiday_Flag'])
joined = joined.dropna()
int_cols = ['Day_Year', 'Off_Flag']
joined[int_cols] = joined[int_cols].applymap(lambda x: int(x))

# create training and testing data sets, generator will separate out the features and target
train_data = joined[joined.index.year != 2017]
# test_data = pd.concat([train_data[-1440:],
#                       joined[joined.index.year == 2017]])
test_data = joined[joined.index.year == 2017]
# standardization for with dummy version
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data.values)
test_data = scaler.transform(test_data.values)

# train_data = train_data[~np.isnan(train_data).any(axis=1)]
# test_data = test_data[~np.isnan(test_data).any(axis=1)]
#######################################

lookback = 336  # 14 days
delay = 0  # how far into the future is the target

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

# regression model  #########################################################################################

def build_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback, joined.shape[-1])))
    # model.add(layers.Flatten(input_shape=(lookback, 1)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit_generator(train_gen,
                              epochs=20,
                              steps_per_epoch=365,
                              validation_data=val_gen,
                              validation_steps=50)

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

test_metrics = model.evaluate_generator(test_gen, steps=10)
test_mae = test_metrics[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[:168*10, 0].mean()))

# test accuracy: 0.64582 (1 step, with dummies, adam opt, reduce on plateau, 80, 40, dropout)
# test accuracy: 0.56213 (3 step, with dummies, adam opt, reduce on plateau, 80, 40, dropout)
# test accuracy: 0.49889 (10 step, with dummies, adam opt, reduce on plateau, 80, 40, dropout)

loss_plot(history=history, skip_epoch=0)
pred_plot(model=model, test=test_gen, test_target=test_data[:, 0], pred_periods=48)
pred_multiplot(model, test_data)

with open(f"models/dense_80-40_wd_do_lr_adam_fin.pickle", "rb") as pfile:
    exec(f"model = pickle.load(pfile)")
with open(f"models/dense_80-40_wd_do_lr_adam_fin_hist.pickle", "rb") as pfile:
    exec(f"history = pickle.load(pfile)")

# the RNN models ####################################################################
import keras.backend as K
import keras.losses


def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

quantile = 0.5

def build_rnn():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu',
                            input_shape=(None, train_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GRU(32))
    # model.add(layers.GRU(32, input_shape=(None, train_data.shape[-1])))
    # model.add(layers.GRU(64, input_shape=(None, 1)))
    model.add(layers.Dense(1))
    # model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.compile(optimizer='rmsprop', loss=lambda y,f: tilted_loss(quantile,y,f), metrics=['mae'])
    return model

model_rnn = build_rnn()
model_rnn.summary()

history_rnn = model_rnn.fit_generator(train_gen,
                                      steps_per_epoch=365,
                                      epochs=20,
                                      validation_data=val_gen,
                                      validation_steps=50,
                                        callbacks = [
                                            callbacks.ReduceLROnPlateau(factor=.5, patience=3, verbose=1)
                                        ]
                                      )

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

test_metrics_rnn = model_rnn.evaluate_generator(test_gen, steps=10)
test_mae = test_metrics_rnn[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[:168*10, 0].mean()))

# final model
# test accuracy: 0.85756 (1 step, with dummies (inc dayofyear), correct steps per epoch, 20 epochs)
# test accuracy: 0.84076 (3 step, with dummies (inc dayofyear), correct steps per epoch, 20 epochs)
# test accuracy: 0.77202 (10 step, with dummies (inc dayofyear), correct steps per epoch, 20 epochs)

# test accuracy: 0.85767 (1 step, with dummies (inc dayofyear), correct steps per epoch, dropout, 2 layer, 40 epochs)
# test accuracy: 0.84749 (3 step, with dummies (inc dayofyear), correct steps per epoch, dropout, 2 layer, 40 epochs)
# test accuracy: 0.77886 (10 step, with dummies (inc dayofyear), correct steps per epoch, dropout, 2 layer, 40 epochs)

loss_plot(history=history_rnn, skip_epoch=0)
pred_plot(model=model_rnn, test=test_gen, test_target=test_data[:, 0], pred_periods=48)
pred_multiplot(model_rnn, test_data)


# with open(f'models/rnn_10_rev.pickle', 'wb') as pfile:
#     pickle.dump(model_rnn, pfile)
# with open(f'models/rnn_10_rev_hist.pickle', 'wb') as pfile:
#     pickle.dump(history_rnn, pfile)
#
# with open(f"models/rnn_20_wd_do_2layer_fin.pickle", "rb") as pfile:
#     exec(f"model_rnn = pickle.load(pfile)")
# with open(f"models/rnn_20_wd_do_2layer_fin_hist.pickle", "rb") as pfile:
#     exec(f"history_rnn = pickle.load(pfile)")
#
with open(f"models/rnn_20_wd_fin.pickle", "rb") as pfile:
    exec(f"model_rnn = pickle.load(pfile)")
with open(f"models/rnn_20_wd_fin_hist.pickle", "rb") as pfile:
    exec(f"history_rnn = pickle.load(pfile)")

with open(f"models/crnn_20_wd_nw_32.pickle", "rb") as pfile:
    exec(f"model_rnn = pickle.load(pfile)")
with open(f"models/crnn_20_wd_nw_32_hist.pickle", "rb") as pfile:
    exec(f"history_rnn = pickle.load(pfile)")

with open(f"models/crnn_20_wd_nw_336lb_2dum_mapeloss.pickle", "rb") as pfile:
    exec(f"model_rnn = pickle.load(pfile)")
with open(f"models/crnn_20_wd_nw_336lb_2dum_mapeloss_hist.pickle", "rb") as pfile:
    exec(f"history_rnn = pickle.load(pfile)")

with open(f"models/crnn_20_wd_nw_336lb_q50.pickle", "rb") as pfile:
    exec(f"model_rnn = pickle.load(pfile)")
with open(f"models/crnn_20_wd_nw_336lb_q50_hist.pickle", "rb") as pfile:
    exec(f"history_rnn = pickle.load(pfile)")

# model_rnn = keras.models.load_model('models/crnn_20_wd_nw_336lb_q50.h5', custom_objects={'tilted_loss':tilted_loss})

with open(f"models/rnn_q50_pred_step1.pickle", "rb") as pfile:
    exec(f"predictions = pickle.load(pfile)")
with open(f"models/rnn_q95_pred_step1.pickle", "rb") as pfile:
    exec(f"predictions95 = pickle.load(pfile)")
with open(f"models/rnn_q05_pred_step1.pickle", "rb") as pfile:
    exec(f"predictions05 = pickle.load(pfile)")

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)
predictions_m = model_rnn.predict_generator(test_gen, steps=1)

def rescale(num_list, pre_scaled_data):
    min_scale = min(pre_scaled_data[pre_scaled_data.index.year != 2017].iloc[:, 0])
    max_scale = max(pre_scaled_data[pre_scaled_data.index.year != 2017].iloc[:, 0])
    return [x * (max_scale - min_scale) + min_scale for x in num_list]

# plt.plot(rescale(predictions_m, joined), label='mean')
plt.clf()
plt.xlabel('Hour')
plt.ylabel('GW')
plt.ylim((7, 16))
plt.plot(rescale(predictions_m.flatten(), joined), '--', c='red', label='prediction')
plt.gca().margins(x=0)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(24))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.gca().fill_between(list(range(168)), rescale(predictions05.flatten(), joined), rescale(predictions95.flatten(), joined),
                       facecolors='b', alpha=0.3)
# plt.plot(rescale(predictions95.flatten(), joined), label='q95')
# plt.plot(rescale(predictions05.flatten(), joined), label='q05')
plt.plot(rescale(test_data[:168, 0], joined), '-', c='black', label='actual')
plt.legend(loc=2)
plt.show()
plt.savefig('quantile_rnn_168.png', dpi=800)


loss_plot(history=history_rnn, skip_epoch=1)

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

test_metrics_rnn = model_rnn.evaluate_generator(test_gen, steps=10)
test_mae = test_metrics_rnn[1]
# using the following accuracy definition
print('test accuracy: {:.5f}'.format(1-test_mae/test_data[1440:(1440+168*10), 0].mean()))

# test accuracy: 0.80151 (1 step, with dummies no weather, cnn + rnn)
# test accuracy: 0.82494 (3 step, with dummies no weather, cnn + rnn)
# test accuracy: 0.81818 (10 step, with dummies no weather, cnn + rnn)

# shifted
# test accuracy: 0.76606 (1 step, with dummies no weather, cnn + rnn)
# test accuracy: 0.80622 (3 step, with dummies no weather, cnn + rnn)
# test accuracy: 0.81057 (10 step, with dummies no weather, cnn + rnn)

# shifted
# test accuracy: 0.75857 (1 step, with dummies no weather, rnn)
# test accuracy: 0.83246 (3 step, with dummies no weather, rnn)
# test accuracy: 0.84447 (10 step, with dummies no weather, rnn)

y_t, y_p, err = pred_plot_per_step_rev(test_data, model_rnn, pre_scaled_data=joined)

pred_plot_per_step(test_data, model_rnn)
pred_plot_per_step_rev(test_data, model_rnn, pre_scaled_data=joined)
pred_plot_per_step(test_data, model_rnn, metric='mae')
mape_rpt(test_data, model_rnn)
pred_multiplot(model_rnn, test_data)

# Average mape over the forecast horizon is 0.246335 (rnn_20_wd_nw_336lb)
# Average mape over the forecast horizon is 0.247127 (rnn_20_wd_nw_336lb_2dum)
# Average mape over the forecast horizon is 0.242175 (crnn_20_wd_nw_336lb_2dum)
# Average mape over the forecast horizon is 0.245870 (crnn_20_wd_nw_336lb_2dum_do)
# Average mape over the forecast horizon is 0.244773 (crnn_20_wd_nw_336lb_2dum_mapeloss)
# Average mape over the forecast horizon is 0.246188 (rnn_20_wd_nw_336lb_bd)
# Average mape over the forecast horizon is 0.251149 (rnn_20_wd_ww_336lb_reg)

# Average mape over the forecast horizon is 0.099116 (crnn_20_wd_nw_336lb_w)
# Average mape over the forecast horizon is 0.121930 (models/crnn_20_wd_nw_336lb_sc)

# test mape 1 step: 0.20159
# test mape 3 step: 0.17606
# test mape 10 step: 0.17760

# single plot prediction vs actual
pred_plot(test_data, model_rnn, pre_scaled_data=joined, steps=[1])  #, savefile=True, savename='crnn_final_wk.png')
pred_plot(test_data, model_rnn, pre_scaled_data=joined, steps=[3])
pred_plot(test_data, model_rnn, pre_scaled_data=joined, steps=[9,12])
pred_plot(test_data, model_rnn, pre_scaled_data=joined, steps=[20,21])
pred_plot(test_data, model_rnn, pre_scaled_data=joined, steps=[32,33])

test_gen = generator(test_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=None)

predictions = model_rnn.predict_generator(test_gen, steps=52)

mape(test_data[:168, 0], predictions[:168].flatten())

plt.plot(predictions, alpha=0.5, label='prediction')
plt.plot(test_data[:len(predictions), 0], alpha=0.5, label='actual')
plt.legend()
plt.show()

# continued from models_classical script (need whole year prophet forecast)
forecast_yr = forecast[forecast['ds'].dt.year==2017]

plt.plot(rescale(test_data[:, 0], joined), label='actual')
plt.plot(forecast_yr['yhat'].tolist(), label='actual')
plt.legend()
plt.show()

df_comp = pd.DataFrame({'fb_pred': forecast_wk['yhat'].values,
                       'rnn_pred': rescale(predictions[:168].flatten(), joined),
                       'actual': rescale(test_data[:168, 0], joined)})
df_comp['fb_actual'] = df_comp['fb_pred'] - df_comp['actual']
df_comp['rnn_actual'] = df_comp['rnn_pred'] - df_comp['actual']

# to rescale (inverse of minmax scaler)
def rescale(num_list, pre_scaled_data):
    min_scale = min(pre_scaled_data[pre_scaled_data.index.year != 2017]['COAST'])
    max_scale = max(pre_scaled_data[pre_scaled_data.index.year != 2017]['COAST'])
    return [x * (max_scale - min_scale) + min_scale for x in num_list]