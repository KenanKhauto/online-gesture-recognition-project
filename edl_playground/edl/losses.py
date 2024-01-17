import math

from typing import Type
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn.functional as F

from .losses_base import Loss


def _type_2_maximum_likelihood_loss(
    y: Tensor,
    alpha: Tensor,
    S: Tensor
) -> Tensor:
    """Type II Maximum Likelihood based on https://arxiv.org/abs/1806.01768

    Args:
        y: one-hot vector encoding the ground-truth (N, K)

        alpha: Parameters of Dirichlet (N, K)

        S: Dirichlet strength (sum of alphas) (N)

    Returns:
        loss (L): Loss (N)
    """
    loss = torch.sum(y * (S.log()[:, None] - alpha.log()), dim=1)
    return loss


def _bayes_risk_for_cross_entropy_loss(
    y: Tensor,
    alpha: Tensor,
    S: Tensor
) -> Tensor:
    """Bayes risk for cross-entropy loss based on https://arxiv.org/abs/1806.01768

    Args:
        y: one-hot vector encoding the ground-truth (N, K)

        alpha: Parameters of Dirichlet (N, K)

        S: Dirichlet strength (sum of alphas) (N)

    Returns:
        loss (L): Loss (N)
    """
    loss = torch.sum(y * (torch.digamma(S)[:, None] - torch.digamma(alpha)), dim=1)
    return loss


def _bayes_risk_for_sse_loss(
    y: Tensor,
    alpha: Tensor,
    S: Tensor
) -> Tensor:
    """Bayes risk for sum of squares loss based on https://arxiv.org/abs/1806.01768

    Args:
        y: one-hot vector encoding the ground-truth (N, K)

        alpha: Parameters of Dirichlet (N, K)

        S: Dirichlet strength (sum of alphas) (N)

    Returns:
        loss (L): loss (N)
    """
    S = S[:, None]
    p = alpha / S # expected probability

    loss = torch.sum((y - p)**2 + p*(1-p)/(S+1), dim=1)
    return loss


def _KL_divergence(
    alpha_tilde: Tensor,
    annealing_coeff: float,
):
    """KL divergence from uniform distribution (total uncertainty) based on https://arxiv.org/abs/1806.01768

    Args:
        alpha_tilde: Parameters of Dirichlet after removal of the non-misleading evidence (N, K)
        annealing_coeff: Annealing coefficient

    Returns:
        divergence: KL divergence (N)

    """
    sum_alpha_tildes = torch.sum(alpha_tilde, dim=1) # (N)
    K = alpha_tilde.shape[1]

    divergence = torch.lgamma(sum_alpha_tildes) - torch.lgamma(torch.tensor(K, dtype=alpha_tilde.dtype)) - torch.sum(torch.lgamma(alpha_tilde), dim=1) \
    + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tildes)[:, None]), dim=1)
    return annealing_coeff * divergence


def _EUC_loss(
    prob: Tensor,
    target: Tensor,
    uncertainty: Tensor,
    annealing_coeff: float,
    eps: float=1e-7,
):
    """Evidential Uncertainty Calibration (EUC) method based on https://arxiv.org/abs/2107.10161

    Args:
        prob: Probabilities for each class (N, K)
        target: Target classes for each sample (N)
        uncertainty: Uncertainty value for each sample (N)
        annealing_coeff: Annealing coefficient

    Returns:
        loss: Returns EUC loss (IC loss + AU loss) (N)

    """
    max_prob = prob.max(dim=1)
    pred = max_prob.indices # (N)
    pred_prob = max_prob.values # (N)
    accurate = pred.eq(target) # (N)
    inaccurate = ~accurate

    uncertain = - annealing_coeff * pred_prob * (1 - uncertainty + eps).log() # (N)
    accurate_uncertain = accurate.to(uncertain.dtype) * uncertain

    certain = - (1 - annealing_coeff) * (1 - pred_prob) * (uncertainty + eps).log() # (N)
    inaccurate_certain = inaccurate.to(certain.dtype) * certain

    loss = accurate_uncertain + inaccurate_certain
    return loss


class AnnealingFactor(ABC):
    @abstractmethod
    def __call__(self, step: int):
        ...


class LinearAnnealingFactor(AnnealingFactor):
    def __init__(self, annealing_steps: int=10):
        self.annealing_steps = annealing_steps
        super().__init__()

    def __call__(self, step: int):
        return min(1.0, step / self.annealing_steps)


class ExpAnnealingFactor(AnnealingFactor):
    def __init__(self, initial_value: float=0.01, annealing_steps: int=10):
        self.initial_value = initial_value
        self.annealing_steps = annealing_steps
        super().__init__()

    def __call__(self, step: int):
        return self.initial_value * math.exp(-math.log(self.initial_value) * min(1.0, step / self.annealing_steps))


class EDL_DataLoss(Loss):
    @abstractmethod
    def _get_loss(self, y: Tensor, alpha: Tensor, S: Tensor) -> Tensor:
        ...

    def __call__(self, evidence: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        K = evidence.shape[1]
        y = F.one_hot(target, K) # one-hot vector encoding the ground-truth (N, K)
        alpha = evidence + 1 # (N, K)
        S = torch.sum(alpha, dim=1) # (N)
        loss = self._get_loss(y, alpha, S)
        return loss


class Type2MaximumLikelihoodLoss(EDL_DataLoss):
    def _get_loss(self, y: Tensor, alpha: Tensor, S: Tensor) -> Tensor:
        loss = _type_2_maximum_likelihood_loss(y, alpha, S)
        return loss


class BayesRiskForCrossEntropyLoss(EDL_DataLoss):
    def _get_loss(self, y: Tensor, alpha: Tensor, S: Tensor) -> Tensor:
        loss = _bayes_risk_for_cross_entropy_loss(y, alpha, S)
        return loss


class BayesRiskForSSELoss(EDL_DataLoss):
    def _get_loss(self, y: Tensor, alpha: Tensor, S: Tensor) -> Tensor:
        loss = _bayes_risk_for_sse_loss(y, alpha, S)
        return loss


EDL_DATALOSSES = {
    "ml": Type2MaximumLikelihoodLoss,
    "ce": BayesRiskForCrossEntropyLoss,
    "sse": BayesRiskForSSELoss,
}


def get_edl_dataloss(loss_name: str) -> Type[EDL_DataLoss]:
    return EDL_DATALOSSES[loss_name.lower()]


class EDL_RegularizationLoss(Loss):
    def __init__(self, annealing_factor: AnnealingFactor):
        self.annealing_factor = annealing_factor
        super().__init__()

    @abstractmethod
    def __call__(self, evidence: Tensor, target: Tensor, training_epoch: int, *args, **kwargs) -> Tensor:
        ...


class EUC_RegularizationLoss(EDL_RegularizationLoss):
    def __init__(self, annealing_factor: AnnealingFactor=ExpAnnealingFactor()):
        super().__init__(annealing_factor)

    def __call__(self, evidence: Tensor, target: Tensor, training_epoch: int, *args, **kwargs) -> Tensor:
        K = evidence.shape[1]
        annealing_coeff = self.annealing_factor(training_epoch)
        _, prob, uncertainty = get_belief_probs_and_uncertainty(evidence, K)
        loss = _EUC_loss(prob, target, uncertainty, annealing_coeff)
        return loss


class KL_Divergence_RegularizationLoss(EDL_RegularizationLoss):
    def __init__(self, annealing_factor: AnnealingFactor=LinearAnnealingFactor()):
        super().__init__(annealing_factor)

    def __call__(self, evidence: Tensor, target: Tensor, training_epoch: int, *args, **kwargs) -> Tensor:
        K = evidence.shape[1]
        y = F.one_hot(target, K) # one-hot vector encoding the ground-truth (N, K)
        alpha = evidence + 1 # (N, K)
        alpha_tilde = y + (1 - y) * alpha
        annealing_coeff = self.annealing_factor(training_epoch)
        loss = _KL_divergence(alpha_tilde, annealing_coeff)
        return loss


EDL_REGULARIZATIONS = {
    "kl": KL_Divergence_RegularizationLoss,
    "euc": EUC_RegularizationLoss,
}


def get_edl_regularization(regularization_name: str) -> Type[EDL_RegularizationLoss]:
    return EDL_REGULARIZATIONS[regularization_name.lower()]


def get_belief_probs_and_uncertainty(evidence, num_classes):
    """Calculates belief mass, expected probabilities and uncertainty for EDL based on https://arxiv.org/abs/1806.01768

    Args:
        evidence (e): evidence vector generated by NN (N, K) = (<#BATCHES>, <#CLASSES>), non-negative (activation already applied to model output)
        
        num_classes (K): Number of classes

    Returns:
        belief (b): Belief mass for each class (N, K)
        probs (p): Expected probability for each class (N, K)
        uncertainty (u): Uncertainty for each Sample (N)
    """
    alpha = evidence + 1 # (N, K)
    S = torch.sum(alpha, dim=1) # (N)
    belief = evidence / S[:, None] # (N, K) / (N) = (N, K)
    probs = alpha / S[:, None] # (N, K) / (N) = (N, K)
    uncertainty = num_classes / S # (1) / (N) = (N)
    return belief, probs, uncertainty


def get_correct_preds(evidence, target, uncertainty_thresh: float, keep_as_tensor: bool=False) -> int:
    """ Calculates the number of correct predictions based on https://arxiv.org/abs/1806.01768

    Args:
        evidence (e): evidence vector generated by NN (N, K) = (<#BATCHES>, <#CLASSES>), non-negative (activation already applied to model output)

        target: target vector containing the correct class for each sample (N) 

        uncertainty_thresh (float): Uncertainty threshold between 0 and 1 above which the model is assumed to reject making predictions (predicts "I do not know")

    Returns:
        correct (int): Number of correctly classified samples considering the uncertainty threshold
        rejected_corrects (int): Number of correctly classified samples rejected due to uncertainty
        correctly_classified (int): Number of correctly classified samples regardless of uncertainty
    """
    K = evidence.shape[1]
    belief, probs, uncertainty = get_belief_probs_and_uncertainty(evidence, K)
    certain_pred = uncertainty[:, None] <= uncertainty_thresh # (N) > (N, 1)
    pred = probs.argmax(dim=1, keepdim=True) # (N, 1)
    correctly_classified = pred.eq(target.view_as(pred))
    correct = (correctly_classified & certain_pred).sum()
    rejected_corrects = (correctly_classified & ~certain_pred).sum()
    correctly_classified = correctly_classified.sum()
    if not keep_as_tensor:
        correct, rejected_corrects, correctly_classified = correct.item(), rejected_corrects.item(), correctly_classified.item()
    return correct, rejected_corrects, correctly_classified