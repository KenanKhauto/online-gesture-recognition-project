import torch
import torch.nn.functional as F

from .losses import get_correct_preds


def test(model, device, test_loader, uncertainty_thresh, print_fn=print):
    model.eval()
    total_correct = 0
    total_rejected_corrects = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            evidence = F.relu(output)
            correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)
            total_correct += correct
            total_rejected_corrects += rejected_corrects
                
    test_acc = total_correct / len(test_loader.dataset)
    share_rejected_corrects = total_rejected_corrects / len(test_loader.dataset)

    print_fn('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_correct, len(test_loader.dataset),
        100. * test_acc))

    return test_acc, share_rejected_corrects