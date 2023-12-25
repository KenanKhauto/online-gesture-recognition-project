import numpy as np
import matplotlib.pyplot as plt


def stats_plot(thresholds, train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects, last_n_epochs: int=10):
    fig, axs = plt.subplots(1, 3, figsize=(20, 16))

    train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects = [np.array(x) for x in [train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects]]

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    axs[2].set_title("Share of Rejected Corrects due to Uncertainty")

    axs[0].set_xlabel("uncertainty thresh"), axs[0].set_ylabel("loss")
    axs[1].set_xlabel("uncertainty thresh"), axs[1].set_ylabel("accuracy")
    axs[2].set_xlabel("uncertainty thresh"), axs[2].set_ylabel("rejected corrects")

    for i, (train, val) in enumerate(zip(
        [train_loss,    train_acc,  train_rejected_corrects ],
        [None,          val_acc,    val_rejected_corrects   ]
    )):
        ax = axs[i]
        ax.plot(thresholds, train, label="train")
        if i != 0:
            ax.plot(thresholds, val, label="val")
        ax.legend()
        ax.ticklabel_format(useOffset=False, style='plain')

    return fig, axs