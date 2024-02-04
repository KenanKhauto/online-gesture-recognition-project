import os
import time
import copy
from typing import List, Dict

import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tnrange
from IPython import display
from ipywidgets import Output

from .train import train
from .test import test
from .plots import stats_plot


def _LD_to_DL(LD: List[dict]) -> Dict[str, list]:
    return {k: [d[k] for d in LD] for k in LD[0]}


def run(epochs, scheduler, display_stats_plot, model, device, train_loader, test_loader, criterion, optimizer, uncertainty_thresh, log_interval, dry_run, checkpoint_dir, save_model, model_name, preceding_epoch, consider_thresh_for_best_model):
    train_metrics = []
    test_metrics = []

    best_model_state = None
    best_model_score = None

    for epoch in (pbar := tnrange(1, epochs + 1)):
        if epoch == 1 and display_stats_plot:
            out = Output()
            display.display(out)

        epoch_train_metrics = train(model, device, train_loader, criterion, optimizer, epoch, uncertainty_thresh, log_interval, dry_run, pbar.set_description)
        epoch_test_metrics = test(model, device, test_loader, uncertainty_thresh, pbar.set_description)

        if consider_thresh_for_best_model:
            model_score = epoch_test_metrics["AccU"]
        else:
            model_score = epoch_test_metrics["Acc"]

        if best_model_score is None:
            pbar.set_description('Epoch: {} \tVal acc {:.2f}% \t Saving model...'.format(
                epoch, 100. * model_score))
        elif model_score > best_model_score:
            pbar.set_description('Epoch: {} \tBetter val acc ({:.2f}% => {:.2f}%) \t Saving model...'.format(
                epoch, 100. * best_model_score, 100. * model_score))
            
        if best_model_score is None or model_score > best_model_score:
            best_model_score = model_score
            best_model_state = {
                "epoch": preceding_epoch + epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            }

            if save_model:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
                torch.save(best_model_state, path)

        train_metrics.append(epoch_train_metrics)
        test_metrics.append(epoch_test_metrics)

        scheduler.step()

        if display_stats_plot:
            train_metrics_DL = _LD_to_DL(train_metrics)
            test_metrics_DL = _LD_to_DL(test_metrics)
            if epoch == epochs:
                out.close()
                fig, axs = stats_plot(train_metrics_DL, test_metrics_DL, uncertainty_thresh)
                plt.show()
            else:
                with out:
                    fig, axs = stats_plot(train_metrics_DL, test_metrics_DL, uncertainty_thresh)
                    plt.show()
                    # display.display(plt.gcf())
                    display.clear_output(wait=True)
                    time.sleep(1)

    elapsed = pbar.format_dict["elapsed"]
    rate = 1 / pbar.format_dict["rate"] # s/it

    elapsed_str = pbar.format_interval(elapsed)
    rate_str = pbar.format_interval(rate)

    best_model = model.load_state_dict(best_model_state["model_state_dict"])

    print(f"Best Val ACC: {100. * best_model_score:.2f}%")
    print(f"Total Runtime: {elapsed_str} ({rate_str} per Epoch for {epochs} Epochs)")

    return best_model, train_metrics, test_metrics


def eval_run(model, device, test_loader, uncertainty_thresh):
    test_metrics = test(model, device, test_loader, uncertainty_thresh, print)
    return test_metrics