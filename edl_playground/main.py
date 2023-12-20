# https://github.com/pytorch/examples/blob/main/mnist/main.py

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

DATASET = "cifar10"

if DATASET == "cifar10":
    from dataloader.cifar10 import get_train_loader, get_test_loader, CLASSES
elif DATASET == "mnist":
    from dataloader.mnist import get_train_loader, get_test_loader, CLASSES

from models.conv_net import ConvNet_MNIST, ConvNet_CIFAR10
from base.utils import get_device
from base.run import run
from edl.losses import EDL_Loss, EDL_LOSSES
from edl.run import run as edl_run


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    print(f"Use CUDA: {use_cuda}")
    print(f"Use MPS: {use_cuda}")

    torch.manual_seed(args.seed)

    device = get_device(use_cuda, use_mps)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = get_train_loader(
        transform=transform,
        subset_first_n_classes=args.classes,
        n_samples=args.train_samples
    )
    test_loader = get_test_loader(
        transform=transform,
        subset_first_n_classes=args.classes,
        n_samples=args.test_samples
    )

    num_of_classes = CLASSES if args.classes is None else args.classes

    if DATASET == "cifar10":
        model = ConvNet_CIFAR10(num_of_classes).to(device)
    elif DATASET == "mnist":
        model = ConvNet_MNIST(num_of_classes).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = EDL_Loss(args.uncertainty_loss) if args.uncertainty else F.nll_loss
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.uncertainty:
        train_loss, train_acc, train_rejected_corrects, test_acc, test_rejected_corrects = edl_run(args.epochs, scheduler, args.stats_plot, model, device, train_loader, test_loader, criterion, optimizer, args.uncertainty_thresh, args.log_interval, args.dry_run)
    else:
        train_loss, train_acc, test_loss, test_acc = run(args.epochs, scheduler, args.stats_plot, model, device, train_loader, test_loader, criterion, optimizer, args.log_interval, args.dry_run)

    if args.save_model:
        torch.save(model.state_dict(), f"{DATASET}_cnn.pt") 

    if args.uncertainty:
        return train_loss, train_acc, train_rejected_corrects, test_acc, test_rejected_corrects
    else:
        return train_loss, train_acc, test_loss, test_acc


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    parser.add_argument('--uncertainty', action='store_true', help='Use uncertainty')
    parser.add_argument('--no-uncertainty', dest='uncertainty', action='store_false', help='Do not use uncertainty')
    parser.set_defaults(uncertainty=True)
    parser.add_argument('--uncertainty_loss', choices=EDL_LOSSES, default="ml",
                        help='Loss function to use when using uncertainty')
    parser.add_argument('--uncertainty-thresh', type=float, default=0.9,
                        help='Uncertainty threshold above which the model is assumed to reject making predictions (predicts "I do not know")')
    
    parser.add_argument('--stats-plot', action='store_true', help='Display stats plot')
    parser.add_argument('--no-stats-plot', dest='stats_plot', action='store_false', help='Do not display stats plot')
    parser.set_defaults(stats_plot=True)

    parser.add_argument('--classes', type=int, default=None, metavar='N',
                        help='Only use the first N classes')
    parser.add_argument('--train-samples', type=int, default=None, metavar='N',
                        help='Only use N train samples per class')
    parser.add_argument('--test-samples', type=int, default=None, metavar='N',
                        help='Only use N test samples per class')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)