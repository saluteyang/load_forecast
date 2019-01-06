# generators for training, validation and test sets
# this uses an offset, parameterized by the delay variable, and is not consistent with the other models
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

# model input data for below follows joined.copy() step
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

predictions = model5.predict_generator(test_gen, steps=55)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), joined_keep[:1000, 0], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model5.evaluate_generator(test_gen, steps=55)

# results: mae on training 0.7444, on validation 1.1637, on test 1.3112 (20 epochs)
# results: mae on training 0.6535, on validation 1.5005, on test 1.6972 (40 epochs)

# model6: add a dense layer *after* the GRU layer, keeping dummies

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

def build_model6():
    model = models.Sequential()
    model.add(layers.GRU(64,  # return_sequences=True,
                         input_shape=(None, joined_keep.shape[-1])))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

model6 = build_model6()
history = model6.fit_generator(train_gen,
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

predictions = model6.predict_generator(test_gen, steps=55)
plt.plot(range(1000), predictions[:1000], 'r', label='test predictions', alpha=0.2)
plt.plot(range(1000), joined_keep[:1000, 0], 'b', label='test actual', alpha=0.2)
plt.xticks(rotation=90)
plt.legend()
plt.show()

model6.evaluate_generator(test_gen, steps=55)

# results: mae on training 0.684, on validation 1.706, on test 1.983 (20 epochs)