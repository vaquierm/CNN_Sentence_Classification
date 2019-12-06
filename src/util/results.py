# This file is a utility file that contains all logic to save results
import matplotlib.pyplot as plt
import os
import numpy as np


def save_training_history(history: dict, acc_img_path: str, loss_img_path: str):
    """
    Save the validation and training accuracy/loss
    :param history: keras model history
    :param acc_img_path: File path to save accuracy image
    :param loss_img_path: File path to save loss image
    """
    if 'acc' in history:
        acc = 'acc'
        val_acc = 'val_acc'
    else:
        acc = 'accuracy'
        val_acc = 'val_accuracy'

    fig_id = int(np.random.uniform(1, 1000, 1)[0])

    plt.figure(fig_id)
    plt.plot(history[acc])
    plt.plot(history[val_acc])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(acc_img_path)

    fig_id += 1

    # Plot training & validation loss values
    plt.figure(fig_id)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(loss_img_path)

    fig_id += 1

def compute_average_histories(histories: list):
    """
    Computes the average of all
    :param histories: List of training histories
    :return: Average training history
    """
    N = len(histories)
    avg_hist = {}
    for k in histories[0].keys():
        temp_buff = np.empty((N, len(histories[0][k])))
        for i in range(N):
            temp_buff[i] = np.array(histories[i][k])
        avg_hist[k] = temp_buff.mean(axis=0).tolist()

    return avg_hist


def save_confusion_matrix(cm: np.ndarray, classes,
                          fig_file_path: str,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Saves the configuration matrix to file
    :param cm: Confusion matrix
    :param classes: List of all classes
    :param fig_file_path: File path to save to
    :param normalize: Normalizes the confusion matrix if true
    :param title: Title of figure
    :param cmap: Color map of confusion matrix
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not os.path.isdir(os.path.dirname(fig_file_path)):
        raise Exception("The directory", os.path.dirname(fig_file_path), "you are trying to save your confusion matrix does not exist")

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylim(len(classes) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(fig_file_path)
