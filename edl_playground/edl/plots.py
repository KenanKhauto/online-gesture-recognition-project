import numpy as np
import matplotlib.pyplot as plt


def _map_layout(layout: list, d: dict) -> list:
    mapped_layout = []
    for e in layout:
        mapped_layout.append(d[e] if e is not None else None)
    return mapped_layout


def stats_plot(train_metrics: dict, val_metrics: dict, uncertainty_thresh: float, last_n_epochs: int=10):
    fig, axs = plt.subplots(2, 4, figsize=(20, 21))

    train_metrics = {k: np.array(v) for k, v in train_metrics.items()}
    val_metrics = {k: np.array(v) for k, v in val_metrics.items()}
    last_train_metrics = {k: v[-last_n_epochs:] for k, v in train_metrics.items()}
    last_val_metrics = {k: v[-last_n_epochs:] for k, v in val_metrics.items()}

    axs[0, 0].set_title("Loss")
    axs[0, 1].set_title(f"AccU (u <= {uncertainty_thresh})")
    axs[0, 2].set_title(f"Acc")
    axs[0, 3].set_title("SRC")
    axs[1, 0].set_title(f"Loss (Last {last_n_epochs} Epochs)")
    axs[1, 1].set_title(f"AccU (u <= {uncertainty_thresh}) (Last {last_n_epochs} Epochs)")
    axs[1, 2].set_title(f"Acc (Last {last_n_epochs} Epochs)")
    axs[1, 3].set_title(f"SRC (Last {last_n_epochs} Epochs)")

    for i in range(2):
        axs[i, 0].set_xlabel("epoch"), axs[i, 0].set_ylabel("loss")
        axs[i, 1].set_xlabel("epoch"), axs[i, 1].set_ylabel("accuracy")
        axs[i, 2].set_xlabel("epoch"), axs[i, 2].set_ylabel("accuracy")
        axs[i, 3].set_xlabel("epoch"), axs[i, 3].set_ylabel("rejected corrects")

    train_layout =  ["loss",    "AccU",  "Acc",  "SRC"]
    val_layout =    [None,      "AccU",  "Acc",  "SRC"]

    for i, (train, val) in enumerate(zip(
        _map_layout(train_layout, train_metrics) + _map_layout(train_layout, last_train_metrics),
        _map_layout(val_layout, val_metrics) + _map_layout(val_layout, last_val_metrics),
    )):
        ax = axs[i//4, i%4]
        ax.plot(train, label="train")
        if i%4 != 0:
            ax.plot(val, label="val")
        ax.legend()
        ax.ticklabel_format(useOffset=False, style='plain')

    return fig, axs