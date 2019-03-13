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

def pred_multiplot(model, test, test_target, steps=10, multi=[1, 3, 5, 10]):
    predictions = model.predict_generator(test, steps=steps)

    f, axarr = plt.subplots(2, 2, sharex=False)

    line1, = axarr[0,0].plot(predictions[:168*multi[0]], label='pred')
    line2, = axarr[0,0].plot(test_target[:168*multi[0], 0], label='actual')
    axarr[0,0].set_title('pred vs actual 1 step')
    axarr[0,0].set_ylabel('load')

    axarr[0,1].plot(predictions[:168*multi[1]], label='pred')
    axarr[0,1].plot(test_target[:168*multi[1], 0], label='actual')
    axarr[0,1].set_title('pred vs actual 3 step')
    axarr[0,1].set_ylabel('load')

    axarr[1,0].plot(predictions[:168*multi[2]], label='pred')
    axarr[1,0].plot(test_target[:168*multi[2], 0], label='actual')
    axarr[1,0].set_title('pred vs actual 5 step')
    axarr[1,0].set_xlabel('periods')
    axarr[1,0].set_ylabel('load')

    axarr[1,1].plot(predictions[:168*multi[3]], label='pred')
    axarr[1,1].plot(test_target[:168*multi[3], 0], label='actual')
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