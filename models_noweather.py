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
cols = [x for x in joined.columns if x!=region]

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

# create training and testing data sets, generator will separate out the features and target
train_data = joined[joined.index.year != 2017]
test_data = joined[joined.index.year == 2017]

# standardization for with dummy version
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data.values)
test_data = scaler.transform(test_data.values)

# train_data = train_data[~np.isnan(train_data).any(axis=1)]
# test_data = test_data[~np.isnan(test_data).any(axis=1)]
#######################################

# testing no dummies version
joined = joined['COAST']

# create training and testing data sets, generator will separate out the features and target
train_data = joined[joined.index.year != 2017]
test_data = joined[joined.index.year == 2017]

# standardization for no dummy version
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_data = scaler.transform(test_data.values.reshape(-1, 1))

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

# val_steps = (60000 - 50001 - lookback)
# test_steps = (len(joined) - 60001 - lookback)

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
# test accuracy: 0.83433 (1 step, no dummies)
# test accuracy: 0.82919 (3 step, no dummies)
# test accuracy: 0.76700 (10 step, no dummies)

# test accuracy: 0.85912 (1 step, with dummies)  -- best performer, day of year doesn't help here (?)
# test accuracy: 0.81311 (3 step, with dummies)
# test accuracy: 0.75873 (10 step, with dummies)

# test accuracy: 0.33058 (1 step, with dummies, earlystop)
# test accuracy: 0.21906 (3 step, with dummies, earlystop)
# test accuracy: 0.17096 (10 step, with dummies, earlystop)

# test accuracy: 0.70826 (1 step, with dummies, correct steps per epoch) - wrong generator after previous model run
# test accuracy: 0.65070 (3 step, with dummies, correct steps per epoch)
# test accuracy: 0.50023 (10 step, with dummies, correct steps per epoch)

# test accuracy: 0.64582 (1 step, with dummies, adam opt, reduce on plateau, 80, 40, dropout)
# test accuracy: 0.56213 (3 step, with dummies, adam opt, reduce on plateau, 80, 40, dropout)
# test accuracy: 0.49889 (10 step, with dummies, adam opt, reduce on plateau, 80, 40, dropout)

loss_plot(history=history, skip_epoch=0)
pred_plot(model=model, test=test_gen, test_target=test_data[:, 0], pred_periods=48)
pred_multiplot(model, test_gen, test_data)

predictions = model.predict_generator(test_gen, steps=52)

err_hist = []
for i in range(52):
    a1, a2 = predictions[i*168: (i+1)*168].flatten(), test_data[i*168: (i+1)*168, 0]
    err_hist.append(mean_abs_err(a1, a2))
plt.plot(err_hist)
plt.show()

# with open(f'models/dense_20_nd.pickle', 'wb') as pfile:
#     pickle.dump(model, pfile)
# with open(f'models/dense_20_nd_hist.pickle', 'wb') as pfile:
#     pickle.dump(history, pfile)

# best performer
# with open(f"models/dense_20_rev.pickle", "rb") as pfile:
#     exec(f"model = pickle.load(pfile)")
# with open(f"models/dense_20_rev_hist.pickle", "rb") as pfile:
#     exec(f"history = pickle.load(pfile)")

with open(f"models/dense_80-40_wd_do_lr_adam_fin.pickle", "rb") as pfile:
    exec(f"model = pickle.load(pfile)")
with open(f"models/dense_80-40_wd_do_lr_adam_fin_hist.pickle", "rb") as pfile:
    exec(f"history = pickle.load(pfile)")

# the RNN models ####################################################################

# early stopping?
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
                                      validation_steps=50
                                      # callbacks=[
                                      #     callbacks.EarlyStopping(patience=5, verbose=1,
                                      #                             restore_best_weights=True)
                                      # ]
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
# test accuracy: 0.83605 (1 step, no dummies)
# test accuracy: 0.84380 (3 step, no dummies)
# test accuracy: 0.79899 (10 step, no dummies)

# test accuracy: 0.85972 (1 step, no dummies, 20 epochs)
# test accuracy: 0.85584 (3 step, no dummies, 20 epochs)
# test accuracy: 0.78466 (10 step, no dummies, 20 epochs)

# test accuracy: 0.85781 (1 step, no dummies, 20 epochs, 2 layers, dropout)
# test accuracy: 0.77481 (10 step, no dummies, 20 epochs, 2 layers, dropout)

# test accuracy: 0.90052 (1 step, with dummies (inc dayofyear), 20 epochs)
# test accuracy: 0.88409 (3 step, with dummies (inc dayofyear), 20 epochs)
# test accuracy: 0.80995 (10 step, with dummies (inc dayofyear), 20 epochs)

# test accuracy: 0.77204 (1 step, with dummies (inc dayofyear), 2 layers, dropout, earlystop)
# test accuracy: 0.75924 (3 step, with dummies (inc dayofyear), 2 layers, dropout, earlystop)
# test accuracy: 0.73952 (10 step, with dummies (inc dayofyear), 2 layers, dropout, earlystop)

# test accuracy: 0.79798 (1 step, with dummies (inc dayofyear), earlystop)
# test accuracy: 0.80139 (3 step, with dummies (inc dayofyear), earlystop)
# test accuracy: 0.77021 (10 step, with dummies (inc dayofyear), earlystop)

# final model
# test accuracy: 0.85756 (1 step, with dummies (inc dayofyear), correct steps per epoch, 20 epochs)
# test accuracy: 0.84076 (3 step, with dummies (inc dayofyear), correct steps per epoch, 20 epochs)
# test accuracy: 0.77202 (10 step, with dummies (inc dayofyear), correct steps per epoch, 20 epochs)

# test accuracy: 0.85767 (1 step, with dummies (inc dayofyear), correct steps per epoch, dropout, 2 layer, 40 epochs)
# test accuracy: 0.84749 (3 step, with dummies (inc dayofyear), correct steps per epoch, dropout, 2 layer, 40 epochs)
# test accuracy: 0.77886 (10 step, with dummies (inc dayofyear), correct steps per epoch, dropout, 2 layer, 40 epochs)

loss_plot(history=history_rnn, skip_epoch=0)
pred_plot(model=model_rnn, test=test_gen, test_target=test_data[:, 0], pred_periods=48)
pred_multiplot(model_rnn, test_gen, test_data)

predictions = model_rnn.predict_generator(test_gen, steps=52)

err_hist_rnn = []
for i in range(52):
    a1, a2 = predictions[i*168: (i+1)*168].flatten(), test_data[i*168: (i+1)*168, 0]
    err_hist_rnn.append(mean_abs_err(a1, a2))
plt.plot(err_hist_rnn)
plt.show()

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

with open(f"centroids.pickle", "rb") as pfile:
    exec(f"centroids = pickle.load(pfile)")

# find assignment to centroids for the weeks
# import operator
# sorted_centroids = sorted(centroids[1].items(), key=operator.itemgetter(1))
assignment = []
week_num = []
for cluster in centroids[1]:
    for i in centroids[1][cluster]:
        week_num.append(i)
        assignment.append(cluster)

pd_centroids = pd.DataFrame({'weeknum': week_num,
                             'cluster': assignment})
pd_centroids.sort_values(by='weeknum', inplace=True)
###
plt.bar(week_num, assignment, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')



