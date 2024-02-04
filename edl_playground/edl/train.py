import torch
import torch.nn.functional as F

from .losses import get_uncertainty_matrix, SCORES


def train(model, device, train_loader, criterion, optimizer, epoch, uncertainty_thresh, log_interval, dry_run, print_fn=print):
    model.train()
    mtx = torch.zeros(4)
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.shape, target.shape, output.shape) # (1, 1, 28, 28), (1), (1, 10)
        # print(data, target, output)
        evidence = F.relu(output)
        individual_loss = criterion(evidence, target, epoch)

        mtx += torch.tensor(get_uncertainty_matrix(evidence, target, uncertainty_thresh))

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

    metrics = {k: v for k, v in zip(["ac", "au", "ic", "iu"], mtx)}
    metrics = {score_fn.__name__: score_fn(*mtx) for score_fn in SCORES}
    metrics["loss"] = train_loss / len(train_loader.dataset)

    return metrics