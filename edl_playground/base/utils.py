import torch


def get_device(use_cuda=True, use_mps=True):
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_correct_preds(output, target) -> int:
    """ Calculates the number of correct predictions

    Args:
        output: output vector of the NN (N, K)

        target: target vector containing the correct class for each sample (N)

    Returns:
        correct (int): Number of correctly classified samples
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct