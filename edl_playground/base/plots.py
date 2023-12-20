import numpy as np
import matplotlib.pyplot as plt


def stats_plot(train_loss, train_acc, val_loss, val_acc, last_n_epochs: int=10):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    train_loss, train_acc, val_loss, val_acc = [np.array(x) for x in [train_loss, train_acc, val_loss, val_acc]]
    last_train_loss, last_train_acc, last_val_loss, last_val_acc = [x[-last_n_epochs:] for x in [train_loss, train_acc, val_loss, val_acc]]

    axs[0, 0].set_title("Loss")
    axs[0, 1].set_title("Accuracy")
    axs[1, 0].set_title(f"Loss (Last {last_n_epochs} Epochs)")
    axs[1, 1].set_title(f"Accuracy (Last {last_n_epochs} Epochs)")

    for i in range(2):
        axs[i, 0].set_xlabel("epoch"), axs[i, 0].set_ylabel("loss")
        axs[i, 1].set_xlabel("epoch"), axs[i, 1].set_ylabel("accuracy")

    for i, (train, val) in enumerate(zip([train_loss, train_acc, last_train_loss, last_train_acc], [val_loss, val_acc, last_val_loss, last_val_acc])):
        ax = axs[i//2, i%2]
        ax.plot(train, label="train")
        ax.plot(val, label="val")	
        ax.legend()
        ax.ticklabel_format(useOffset=False, style='plain')

    return fig, axs