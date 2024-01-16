import torch
import torch.nn.functional as F

from .utils import get_correct_preds


def test(model, device, test_loader, criterion, print_fn=print):
    model.eval()
    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            output = F.log_softmax(output, dim=1)
            loss = torch.sum(criterion(output, target)).item()
            test_loss += loss
            total_correct += get_correct_preds(output, target)

    test_acc = total_correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print_fn('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, total_correct, len(test_loader.dataset),
        100. * test_acc))
    return test_acc, test_loss