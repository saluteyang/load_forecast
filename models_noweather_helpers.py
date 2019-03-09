import matplotlib.pyplot as plt
import numpy as np

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