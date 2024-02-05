import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric


from .losses import AvU, AccU, Acc, get_uncertainties_for_correct_and_incorrect


class ModelUncertaintyDistribution(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=[], dist_reduce_fx="cat")
        self.add_state("incorrect", default=[], dist_reduce_fx="cat")

    def update(self, evidence: Tensor, target: Tensor) -> None:
        uncertainties_correct, uncertainties_incorrect = get_uncertainties_for_correct_and_incorrect(evidence, target)
        self.correct.append(uncertainties_correct)
        self.incorrect.append(uncertainties_incorrect)

    def compute(self) -> Tensor:
        correct = torch.cat(self.correct)
        incorrect = torch.cat(self.incorrect)
        return correct, incorrect

    def plot(self, uncertainty_thresh, bins: int=100, optimal_threshold_precision: int=1000):
        correct, incorrect = self.compute()

        fig, axs = plot_model_uncertainty_distribution(
            correct.numpy(), incorrect.numpy(), uncertainty_thresh,
            bins=bins, optimal_threshold_precision=optimal_threshold_precision,
        )
        return fig, axs


def plot_model_uncertainty_distribution(correct, incorrect, uncertainty_thresh, bins: int=100, optimal_threshold_precision: int=1000, density: bool=True, figsize=None):
    import decimal
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model = UncertaintyDistributionModel(correct, incorrect, bins, optimal_threshold_precision, density)
    density_uncertainty_mtx = model.get_density_uncertainty_matrix
    optimal_AuC_thresh = model.get_optimal_AuC_thresh
    uncertainty_mtx = model.get_uncertainty_matrix
    optimal_thresh = model.get_optimal_thresh
    incorrect_pdf = model.incorrect_pdf
    correct_pdf = model.correct_pdf
    score_fns = [AvU, AccU, Acc]

    thresholds = {"Specified": uncertainty_thresh}
    threshold_metrics = {"Specified": uncertainty_mtx(uncertainty_thresh)}
    threshold_auc_metrics = {"Specified": density_uncertainty_mtx(uncertainty_thresh)}

    name = f"AuC-optimized"
    thresh, mtx, orientation = optimal_AuC_thresh()
    thresholds[name] = thresh
    threshold_metrics[name] = uncertainty_mtx(thresh)
    threshold_auc_metrics[name] = mtx

    for score_fn in score_fns[:-1]:
        name = f"{score_fn.__name__}-optimized"
        thresh = optimal_thresh(score_fn)
        if thresh is None:
            print("HERE")
        thresholds[name] = thresh
        threshold_metrics[name] = uncertainty_mtx(thresh)
        threshold_auc_metrics[name] = density_uncertainty_mtx(thresh)

    X = np.linspace(0, 1, bins+1)

    fig, axs = plt.subplots(1, len(thresholds), figsize=figsize if figsize else (8*len(thresholds), 8))

    digits_of_thresh = max(-decimal.Decimal(str(uncertainty_thresh)).as_tuple().exponent, 3)

    for i, (name, thresh) in enumerate(thresholds.items()):
        ax = axs[i]
        ax.plot(X, correct_pdf(X), color="#2BCDC1" ,label="Correct")
        ax.plot(X, incorrect_pdf(X), color="#F66095", label="Incorrect")
        ax.axvline(x=thresh, color="#393E46", label=f"Threshold {round(thresh, digits_of_thresh)}")

        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Density" if density else "Samples")

        ac, au, ic, iu = threshold_metrics[name]
        ac_auc, au_auc, ic_auc, iu_auc = threshold_auc_metrics[name]

        avu, accu, acc = [score_fn(ac, au, ic, iu) for score_fn in score_fns]
        avu_auc, accu_auc, acc_auc = [score_fn(ac_auc, au_auc, ic_auc, iu_auc) for score_fn in score_fns]

        handles, labels = ax.get_legend_handles_labels()
        patches = [
            mpatches.Patch(color="grey", label=f"AvU: {round(100*avu, 2)} (AuC {round(100*avu_auc, 2)}%)"),
            mpatches.Patch(color="grey", label=f"AccU / Acc: {round(100*accu, 2)} / {round(100*acc, 2)} (AuC {round(100*accu_auc, 2)}% / {round(100*acc_auc, 2)}%)"),
            mpatches.Patch(color="#2BCDC1", label=f"AC: {ac} (AuC {round(100*ac_auc, 2)}%)"),
            mpatches.Patch(color="#2BCDC1", label=f"IU: {iu} (AuC {round(100*iu_auc, 2)}%)"),
            mpatches.Patch(color="#F66095", label=f"IC: {ic} (AuC {round(100*ic_auc, 2)}%)"),
            mpatches.Patch(color="#F66095", label=f"AU: {au} (AuC {round(100*au_auc, 2)}%)"),
        ]
        handles.extend(patches)
        ax.legend(handles=handles, loc="upper center")
        ax.title.set_text(f"{name} Threshold")

    if orientation == 1:
        fig.suptitle("Model Uncertainty Distribution - WRONG ORIENTATION LEARNT!")
    else:
        fig.suptitle("Model Uncertainty Distribution")

    return fig, axs


class UncertaintyDistributionModel:
    def __init__(self, correct, incorrect, bins: int=100, optimal_threshold_precision: int=1000, density: bool=True):
        import scipy.stats
        self.correct = correct
        self.incorrect = incorrect
        self.optimal_threshold_precision = optimal_threshold_precision
        self.density = density

        correct_hist = np.histogram(correct, bins=bins, range=(0,1), density=False)
        self.correct_hist_dist = scipy.stats.rv_histogram(correct_hist, density=False)

        incorrect_hist = np.histogram(incorrect, bins=bins, range=(0,1), density=False)
        self.incorrect_hist_dist = scipy.stats.rv_histogram(incorrect_hist, density=False)

    def get_density_uncertainty_matrix(self, thresh):
        ac = self.correct_hist_dist.cdf(thresh)
        ic = self.incorrect_hist_dist.cdf(thresh)
        au = 1-ac
        iu = 1-ic
        return ac, au, ic, iu
    
    def get_uncertainty_matrix(self, thresh):
        return get_uncertainty_matrix(self.correct, self.incorrect, thresh)
    
    def get_optimal_thresh(self, score_fn):
        return get_optimal_thresh(self.correct, self.incorrect, score_fn)
    
    def get_optimal_AuC_thresh(self):
        return get_optimal_AuC_thresh(self.correct_hist_dist, self.incorrect_hist_dist, self.optimal_threshold_precision)
    
    def correct_pdf(self, X):
        if self.density:
            return self.correct_hist_dist.pdf(X)/len(X)
        return self.correct_hist_dist.pdf(X)
    
    def incorrect_pdf(self, X):
        if self.density:
            return self.incorrect_hist_dist.pdf(X)/len(X)
        return self.incorrect_hist_dist.pdf(X)


def get_uncertainty_matrix(correct, incorrect, thresh):
    ac = np.count_nonzero(correct <= thresh)
    iu = np.count_nonzero(incorrect > thresh)
    au = len(correct) - ac
    ic = len(incorrect) - iu
    return ac, au, ic, iu


def get_optimal_thresh(correct, incorrect, score_fn=AvU):
    values = np.concatenate([correct, incorrect, np.array([0, 1])])
    unique = np.unique(values)
    best_score = None
    best_thresh = None
    for thresh in unique:
        mtx = get_uncertainty_matrix(correct, incorrect, thresh)
        score = score_fn(*mtx)

        if best_score is None or score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh


def get_optimal_AuC_thresh(correct_hist_dist, incorrect_hist_dist, optimal_threshold_precision: int=1000):
    X = np.linspace(0, 1, optimal_threshold_precision+1)

    correct_AuCs = correct_hist_dist.cdf(X)
    incorrect_AuCs = 1-incorrect_hist_dist.cdf(X)

    AuCs = np.vstack([correct_AuCs, incorrect_AuCs])

    AuCs = np.repeat(AuCs[None], 2, axis=0)
    AuCs[1] = 1-AuCs[1]

    # axis 0:   0: cdf_correct, 1-cdf_incorrect     1: 1-cdf_correct, cdf_incorrect
    #               = intended orientation              = wrong orientation (correct if high uncertainty)
    # axis 1:   0: correct                          1: incorrect
    # axis 2:   uncertainty_thresh

    total_correct = AuCs.sum(axis=1)

    # axis 0:   0: cdf_correct, 1-cdf_incorrect     1: 1-cdf_correct, cdf_incorrect
    # axis 1:   uncertainty_thresh

    orientation, thresh_opt_ind = np.unravel_index(np.argmax(total_correct, axis=None), total_correct.shape) # (axis 0, axis 1)

    thresh_opt = thresh_opt_ind / len(X)

    max_AuCs = AuCs[:, :, thresh_opt_ind]

    ac = max_AuCs[orientation, 0]
    iu = max_AuCs[orientation, 1]
    au = 1-ac
    ic = 1-iu
    return thresh_opt, (ac, au, ic, iu), orientation
