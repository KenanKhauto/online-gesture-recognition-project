import torch
import torch.nn.functional as F

Tensor = torch.Tensor


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
    loss = torch.sum(y * (S.log() - alpha.log()), dim=1)
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
    loss = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
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
    p = alpha / S # expected probability

    loss = torch.sum((y - p)**2 + p(1-p)/(S+1), dim=1)
    return loss


def _KL_divergence(alpha_tilde):
    """KL divergence from uniform distribution (total uncertainty) based on https://arxiv.org/abs/1806.01768

    Args:
        alpha_tilde: Parameters of Dirichlet after removal of the non-misleading evidence (N, K)

    Returns:
        divergence: KL divergence (N)

    """
    sum_alpha_tildes = torch.sum(alpha_tilde, dim=1) # (N)
    K = alpha_tilde.shape[1]

    divergence = torch.lgamma(sum_alpha_tildes) - torch.lgamma(K) - torch.sum(torch.lgamma(alpha_tilde), dim=1) \
    + torch.sum((alpha_tilde - 1)(torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tildes)))
    return divergence


def _edl_loss(
    loss_func,
    training_epoch: int,
    y: Tensor,
    alpha: Tensor,
    S: Tensor,
    alpha_tilde: Tensor
) -> Tensor:
    """Bayes risk for sum of squares loss based on https://arxiv.org/abs/1806.01768

    Args:
        loss_func: Loss function with Signature (y, alpha, S) -> loss (N)

        training_epoch (int)

        y: one-hot vector encoding the ground-truth (N, K)

        alpha: Parameters of Dirichlet (N, K)

        S: Dirichlet strength (sum of alphas) (N)

        alpha_tilde: Parameters of Dirichlet after removal of the non-misleading evidence (N, K)

    Returns:
        loss (L): Loss (1)
    """
    annealing_coeff = min(1, training_epoch/10)

    loss = torch.sum(loss_func(y, alpha, S)) + annealing_coeff * torch.sum(_KL_divergence(alpha_tilde))
    return loss


EDL_LOSSES = {
    "ml": _type_2_maximum_likelihood_loss,
    "ce": _bayes_risk_for_cross_entropy_loss,
    "sse": _bayes_risk_for_sse_loss,
}


class EDL_Loss:
    def __init__(self, loss: str):
        """
        Args:
            loss to use (str): Name of the loss to use (either of ml, ce, sse)
        """
        loss_lower = loss.lower()
        if loss_lower in EDL_LOSSES:
            self.loss_func = EDL_LOSSES[loss_lower]
        else:
            raise ValueError(f"{loss} is not in the available loss functions {EDL_LOSSES.keys()}")

    def __call__(self, input, target, training_epoch):
        """Calculates the EDL loss based on inputs and targets

        Args:
            input (evidence): evidence vector generated by NN (N, K) = (<#BATCHES>, <#CLASSES>)
        
            target (y): ground-truth (N)

            training_epoch (int)
        """
        K = input.shape[1]
        evidence = input # (N, K)
        alpha = evidence + 1 # (N, K)
        y = F.one_hot(target, K) # one-hot vector encoding the ground-truth (N, K)

        S = torch.sum(alpha) # (N)
        alpha_tilde = y + (1 - y) * alpha

        loss = _edl_loss(self.loss_func, training_epoch, y, alpha, S, alpha_tilde)
        return loss

# belief_mass = evidence / S # (N, K) / (N) = (N, K)
# uncertainty_mass = K / S # (1) / (N) = (N)

# probs = alpha / S # belief_mass or probs?

