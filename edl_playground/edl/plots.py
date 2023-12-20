import numpy as np
import matplotlib.pyplot as plt


def stats_plot(train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects, last_n_epochs: int=10):
    fig, axs = plt.subplots(2, 3, figsize=(20, 16))

    train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects = [np.array(x) for x in [train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects]]
    last_train_loss, last_train_acc, last_train_rejected_corrects, last_val_acc, last_val_rejected_corrects = [x[-last_n_epochs:] for x in [train_loss, train_acc, train_rejected_corrects, val_acc, val_rejected_corrects]]

    axs[0, 0].set_title("Loss")
    axs[0, 1].set_title("Accuracy")
    axs[0, 2].set_title("Share of Rejected Corrects due to Uncertainty")
    axs[1, 0].set_title(f"Loss (Last {last_n_epochs} Epochs)")
    axs[1, 1].set_title(f"Accuracy (Last {last_n_epochs} Epochs)")
    axs[1, 2].set_title(f"Share of Rejected Corrects (Last {last_n_epochs} Epochs)")

    for i in range(2):
        axs[i, 0].set_xlabel("epoch"), axs[i, 0].set_ylabel("loss")
        axs[i, 1].set_xlabel("epoch"), axs[i, 1].set_ylabel("accuracy")
        axs[i, 2].set_xlabel("epoch"), axs[i, 1].set_ylabel("rejected corrects")

    for i, (train, val) in enumerate(zip(
        [train_loss,    train_acc,  train_rejected_corrects,    last_train_loss,    last_train_acc, last_train_rejected_corrects],
        [None,          val_acc,    val_rejected_corrects,      None,               last_val_acc,   last_val_rejected_corrects  ]
    )):
        ax = axs[i//3, i%3]
        ax.plot(train, label="train")
        if i%3 != 0:
            ax.plot(val, label="val")
        ax.legend()
        ax.ticklabel_format(useOffset=False, style='plain')

    return fig, axs