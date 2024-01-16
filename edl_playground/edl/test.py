import torch
import torch.nn.functional as F

from .losses import get_correct_preds


def test(model, device, test_loader, uncertainty_thresh, print_fn=print):
    model.eval()
    total_correct_considering_thresh = 0
    total_rejected_corrects = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            evidence = F.relu(output)
            correct_considering_thresh, rejected_corrects, correct = get_correct_preds(evidence, target, uncertainty_thresh)
            total_correct_considering_thresh += correct_considering_thresh
            total_rejected_corrects += rejected_corrects
            total_correct += correct

    metrics = {
        "acc_with_thresh": total_correct_considering_thresh / len(test_loader.dataset),
        "acc": total_correct / len(test_loader.dataset),
        "share_rejected_corrects": total_rejected_corrects / len(test_loader.dataset)
    }

    print_fn('\nTest set: Accuracy (u <= {}): {}/{} ({:.2f}%)\n\tAccuracy (regardless of u): {}/{} ({:.2f}%)\n'.format(
        uncertainty_thresh, total_correct_considering_thresh, len(test_loader.dataset), 100. * metrics["acc_with_thresh"],
        total_correct, len(test_loader.dataset), 100. * metrics["acc"],))
    return metrics