import matplotlib.pyplot as plt

import io
import time
from tqdm.notebook import tnrange
from IPython import display
from ipywidgets import Output, Textarea

from .train import train
from .test import test
from .plots import stats_plot


def run(epochs, scheduler, display_stats_plot, model, device, train_loader, test_loader, criterion, optimizer, log_interval, dry_run):
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    for epoch in (pbar := tnrange(1, epochs + 1)):
        if epoch == 1 and display_stats_plot:
            out = Output()
            display.display(out)

        train_acc_, train_loss_ = train(model, device, train_loader, criterion, optimizer, epoch, log_interval, dry_run, pbar.set_description)
        test_acc_, test_loss_ = test(model, device, test_loader, criterion, pbar.set_description)

        train_acc.append(train_acc_)
        test_acc.append(test_acc_)
        train_loss.append(train_loss_)
        test_loss.append(test_loss_)

        scheduler.step()

        if display_stats_plot:
            if epoch == epochs:
                out.close()
                fig, axs = stats_plot(train_loss, train_acc, test_loss, test_acc)
                plt.show()
            else:
                with out:
                    fig, axs = stats_plot(train_loss, train_acc, test_loss, test_acc)
                    plt.show()
                    # display.display(plt.gcf())
                    display.clear_output(wait=True)
                    time.sleep(1)

    elapsed = pbar.format_dict["elapsed"]
    rate = 1 / pbar.format_dict["rate"] # s/it

    elapsed_str = pbar.format_interval(elapsed)
    rate_str = pbar.format_interval(rate)

    print(f"Total Runtime: {elapsed_str} ({rate_str} per Epoch for {epochs} Epochs)")

    return train_loss, train_acc, test_loss, test_acc