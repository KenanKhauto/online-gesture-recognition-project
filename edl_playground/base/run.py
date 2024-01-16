import os
import time
import copy
import matplotlib.pyplot as plt

import torch
from tqdm.notebook import tnrange
from IPython import display
from ipywidgets import Output

from .train import train
from .test import test
from .plots import stats_plot


def run(epochs, scheduler, display_stats_plot, model, device, train_loader, test_loader, criterion, optimizer, log_interval, dry_run, checkpoint_dir, save_model, model_name, preceding_epoch):
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    best_model_state = None
    best_test_acc = None

    for epoch in (pbar := tnrange(1, epochs + 1)):
        if epoch == 1 and display_stats_plot:
            out = Output()
            display.display(out)

        train_acc_, train_loss_ = train(model, device, train_loader, criterion, optimizer, epoch, log_interval, dry_run, pbar.set_description)
        test_acc_, test_loss_ = test(model, device, test_loader, criterion, pbar.set_description)

        if best_test_acc is None:
            pbar.set_description('Epoch: {} \tVal acc {:.0f}% \t Saving model...'.format(
                epoch, 100. * test_acc_))
        elif test_acc_ > best_test_acc:
            pbar.set_description('Epoch: {} \tBetter val acc ({:.0f}% => {:.0f}%) \t Saving model...'.format(
                epoch, 100. * best_test_acc, 100. * test_acc_))
            
        if best_test_acc is None or test_acc_ > best_test_acc:
            best_test_acc = test_acc_
            best_model_state = {
                "epoch": preceding_epoch + epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            }

            if save_model:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
                torch.save(best_model_state, path)

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

    best_model = model.load_state_dict(best_model_state["model_state_dict"])

    print(f"Best Val ACC: {100. * best_test_acc:.0f}%")

    print(f"Total Runtime: {elapsed_str} ({rate_str} per Epoch for {epochs} Epochs)")

    return best_model, train_loss, train_acc, test_loss, test_acc


def eval_run(model, device, test_loader, criterion):
    test_acc, test_rejected_corrects = test(model, device, test_loader, criterion, print)
    return test_acc, test_rejected_corrects