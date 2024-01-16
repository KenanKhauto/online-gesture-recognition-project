import torch
import torch.nn.functional as F

from .losses import get_correct_preds


def train(model, device, train_loader, criterion, optimizer, epoch, uncertainty_thresh, log_interval, dry_run, print_fn=print):
    model.train()
    total_correct_considering_thresh = 0
    total_rejected_corrects = 0
    total_correct = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.shape, target.shape, output.shape) # (1, 1, 28, 28), (1), (1, 10)
        # print(data, target, output)
        evidence = F.relu(output)
        individual_loss = criterion(evidence, target, epoch)

        correct_considering_thresh, rejected_corrects, correct = get_correct_preds(evidence, target, uncertainty_thresh)
        total_correct_considering_thresh += correct_considering_thresh
        total_rejected_corrects += rejected_corrects
        total_correct += correct

        loss = torch.mean(individual_loss)
        loss.backward()
        optimizer.step()

        train_loss += torch.sum(individual_loss.detach()).item()

        if batch_idx % log_interval == 0:
            print_fn('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

    metrics = {
        "acc_with_thresh": total_correct_considering_thresh / len(train_loader.dataset),
        "acc": total_correct / len(train_loader.dataset),
        "share_rejected_corrects": total_rejected_corrects / len(train_loader.dataset),
        "loss": train_loss / len(train_loader.dataset)
    }
    return metrics