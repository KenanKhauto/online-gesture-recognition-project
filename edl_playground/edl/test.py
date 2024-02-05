import torch
import torch.nn.functional as F

from .losses import get_uncertainty_matrix, SCORES


def test(model, device, test_loader, uncertainty_thresh, print_fn=print):
    model.eval()
    mtx = torch.zeros(4)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            evidence = F.relu(output)
            mtx += torch.tensor(get_uncertainty_matrix(evidence, target, uncertainty_thresh))

    metrics = {k: v for k, v in zip(["ac", "au", "ic", "iu"], mtx)}
    metrics = {score_fn.__name__: score_fn(*mtx) for score_fn in SCORES}
    metrics

    correctly_classified = metrics["ac"] + metrics["au"]

    print_fn('\nTest set: AccU (u <= {}): {}/{} ({:.2f}%)\n\tAcc: {}/{} ({:.2f}%)\n'.format(
        uncertainty_thresh, metrics["ac"], len(test_loader.dataset), 100. * metrics["ac"] / len(test_loader.dataset),
        correctly_classified, len(test_loader.dataset), 100. * correctly_classified / len(test_loader.dataset)))
    return metrics