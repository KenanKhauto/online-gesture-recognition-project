import torch
import torch.nn.functional as F

from .losses import get_correct_preds


def train(model, device, train_loader, criterion, optimizer, epoch, uncertainty_thresh, log_interval, dry_run, print_fn=print):
    model.train()
    train_loss = 0
    total_correct = 0
    total_rejected_corrects = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.shape, target.shape, output.shape) # (1, 1, 28, 28), (1), (1, 10)
        # print(data, target, output)
        evidence = F.relu(output)
        individual_loss = criterion(evidence, target, epoch, reduction="none")
        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)
        total_correct += correct
        total_rejected_corrects += rejected_corrects
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

    train_acc = total_correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    share_rejected_corrects = total_rejected_corrects / len(train_loader.dataset)
    return train_acc, train_loss, share_rejected_corrects