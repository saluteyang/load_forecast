import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import statsmodels.tsa.api as smt
import seaborn as sns

def loss_pred_plots(history, skip_epoch, model, test, test_target, pred_periods):
    # prepare the data for train vs validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # get the data for predictions based on model
    predictions = model.predict_generator(test, steps=1)

#     skip_epoch = 2
#     pred_periods = 500
#     test_target = joined['COAST'][60001:]

    f, axarr = plt.subplots(2, sharex=False)

    axarr[0].plot(epochs[skip_epoch:], loss[skip_epoch:], 'r', label='training loss')
    axarr[0].plot(epochs[skip_epoch:], val_loss[skip_epoch:], 'b', label='validation loss')
    axarr[0].set_title('training and validation loss')
    axarr[0].set_xlabel('Epochs')
    axarr[0].set_ylabel('Loss')
    axarr[0].legend()

    axarr[1].plot(range(pred_periods), predictions[:pred_periods], 'r', label='test predictions', alpha=0.2)
    axarr[1].plot(range(pred_periods), test_target[:pred_periods], 'b', label='test actual', alpha=0.2)
    axarr[1].set_title(f'first {pred_periods} periods predictions vs actual')
    axarr[1].tick_params(axis='x', rotation=70)
    # axarr[1].xticks(rotation=90)
    axarr[1].legend()

    f.subplots_adjust(hspace=0.7)
    plt.show()

def loss_plot(history, skip_epoch=0):
    # prepare the data for train vs validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    plt.plot(epochs[skip_epoch:], loss[skip_epoch:], 'r', label='training loss')
    plt.plot(epochs[skip_epoch:], val_loss[skip_epoch:], 'b', label='validation loss')
    plt.title('training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def pred_plot(model, test, test_target, pred_periods):

    # get the data for predictions based on model
    predictions = model.predict_generator(test, steps=1)

    plt.plot(range(pred_periods), predictions[:pred_periods], 'r', label='test predictions')
    plt.plot(range(pred_periods), test_target[:pred_periods], 'b', label='test actual')
    plt.title(f'first {pred_periods} periods predictions vs actual')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def pred_multiplot(model, test_data, test_target, steps=10, multi=[1, 3, 5, 10],
                   lookback=336, delay=0):
    test_gen = generator(test_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=0,
                         max_index=None)
    predictions = model.predict_generator(test_gen, steps=steps)

    f, axarr = plt.subplots(2, 2, sharex=False)

    line1, = axarr[0,0].plot(predictions[:168*multi[0]], label='pred', alpha=0.5)
    line2, = axarr[0,0].plot(test_target[:168*multi[0], 0], label='actual', alpha=0.5)
    axarr[0,0].set_title('pred vs actual 1 step')
    axarr[0,0].set_ylabel('load')

    axarr[0,1].plot(predictions[:168*multi[1]], label='pred', alpha=0.5)
    axarr[0,1].plot(test_target[:168*multi[1], 0], label='actual', alpha=0.5)
    axarr[0,1].set_title('pred vs actual 3 step')
    axarr[0,1].set_ylabel('load')

    axarr[1,0].plot(predictions[:168*multi[2]], label='pred', alpha=0.5)
    axarr[1,0].plot(test_target[:168*multi[2], 0], label='actual', alpha=0.5)
    axarr[1,0].set_title('pred vs actual 5 step')
    axarr[1,0].set_xlabel('periods')
    axarr[1,0].set_ylabel('load')

    axarr[1,1].plot(predictions[:168*multi[3]], label='pred', alpha=0.5)
    axarr[1,1].plot(test_target[:168*multi[3], 0], label='actual', alpha=0.5)
    axarr[1,1].set_title('pred vs actual 10 step')
    axarr[1,1].set_xlabel('periods')
    axarr[1,1].set_ylabel('load')

    plt.figlegend([line1, line2], ['pred', 'actual'], 'lower center')

    f.subplots_adjust(hspace=0.7)
    plt.show()


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

# optional plots for time series EDA
# visualizing load
def load_tsplots(data, colname, periods):
    subsample = data[colname][:periods]

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
def load_temp_distplot(data, temp='Drybulb', load='COAST_Hourly'):
    # data should include temp and load info (hourly)
    plt.figure()
    gspec = gs.GridSpec(3, 3)
    top_hist = plt.subplot(gspec[0, 1:]) # index position starts with 0
    side_hist = plt.subplot(gspec[1:, 0])
    lower_right = plt.subplot(gspec[1:, 1:])

    top_hist.hist(data[temp], normed=True)
    side_hist.hist(data[load], bins=50, orientation='horizontal', normed=True)
    side_hist.invert_xaxis()
    lower_right.scatter(data[temp], data[load])
    lower_right.set_xlabel('temperature (F)')
    lower_right.set_ylabel('load (GW)')
    plt.show()

# calculate test accuracy at every forecast step and plot
def mean_abs_err(a1, a2):
    if len(a1) != len(a2):
        raise ValueError("two series don't have the same length")
    sum_err = 0
    for i in range(len(a1)):
        sum_err = sum([abs(x) for x in np.subtract(a1, a2)])/len(a1)
    return sum_err

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(np.subtract(y_true, y_pred)/y_true))

# plot metric over prediction horizon per step
def pred_plot_per_step(test_data, model, steps=52, lookback=336, delay=0, metric='mape'):
    test_gen = generator(test_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=0,
                         max_index=None)

    predictions = model.predict_generator(test_gen, steps=steps)

    err_hist = []
    y_t = []
    y_p = []
    if metric == 'mae':
        for i in range(steps):
            a1, a2 = test_data[(0+i*168): (0+(i+1)*168), 0], predictions[i*168: (i+1)*168].flatten()
            err_hist.append(mean_abs_err(a1, a2))
            y_t.append(a1)
            y_p.append(a2)
    elif metric == 'mape':
        for i in range(steps):
            a1, a2 = test_data[(0+i*168): (0+(i+1)*168), 0], predictions[i*168: (i+1)*168].flatten()
            err_hist.append(mape(a1, a2))
            y_t.append(a1)
            y_p.append(a2)
    else:
        print('This metric is not defined')

    print('Average {} over the forecast horizon is {:5f}'.format(metric, np.mean(err_hist)))
    plt.plot(err_hist)
    plt.show()
    # return y_t, y_p, err_hist


# calculate MAPE for specified forecast horizons
def mape_rpt(test_data, model, steps=[1, 3, 10], lookback=336, delay=0):
    test_gen = generator(test_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=0,
                         max_index=None)
    max_step = max(steps)
    predictions = model.predict_generator(test_gen, steps=max_step)
    # print(len(predictions))
    print('test mape {} step: {:.5f}'.format(steps[0],
                                             mape(test_data[0:(0+steps[0]*168), 0],
                                                  predictions[:168*steps[0]].flatten())))
    print('test mape {} step: {:.5f}'.format(steps[1],
                                             mape(test_data[0:(0 + steps[1] * 168), 0],
                                                  predictions[:168 * steps[1]].flatten())))
    print('test mape {} step: {:.5f}'.format(steps[2],
                                             mape(test_data[0:(0 + steps[2] * 168), 0],
                                                  predictions[:168 * steps[2]].flatten())))


def pred_plot(test_data, model, savename=None, savefile=False,
               lookback=336, delay=0, pre_scaled_data=None, steps=[1]):
    test_gen = generator(test_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=0,
                         max_index=None)

    predictions = model.predict_generator(test_gen, steps=max(steps))

    # to rescale the predictions
    def rescale(num_list):
        min_scale = min(pre_scaled_data[pre_scaled_data.index.year != 2017]['COAST'])
        max_scale = max(pre_scaled_data[pre_scaled_data.index.year != 2017]['COAST'])
        return [x * (max_scale - min_scale) + min_scale for x in num_list]

    if len(steps) == 1:
        pred_to_plot = rescale(predictions[:168 * steps])
        actual_to_plot = rescale(test_data[0:(0 + 168 * steps), 0])
    elif len(steps) == 2:
        pred_to_plot = rescale(predictions[168 * steps[0]:168 * steps[1]])
        actual_to_plot = rescale(test_data[(0 + 168 * steps[0]):(0 + 168 * steps[1]), 0])
    else:
        print('steps need to be increasing sequence of 2 numbers')

    plt.clf()
    plt.plot(pred_to_plot, '--', c='red', label='prediction')
    plt.plot(actual_to_plot, '-', c='black', label='actual')
    plt.xlabel('Hour')
    plt.ylabel('GW')
    plt.legend(loc=2)
    if savefile:
        plt.savefig(savename, dpi=800)
    else:
        plt.show()


