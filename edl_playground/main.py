# https://github.com/pytorch/examples/blob/main/mnist/main.py

import argparse
from typing import Tuple

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import Module

from base.utils import get_device
from base.run import run, eval_run
from base.losses import NLL_Loss, Loss
from edl.losses import (
    EDL_DATALOSSES, EDL_REGULARIZATIONS,
    Type2MaximumLikelihoodLoss, BayesRiskForCrossEntropyLoss, BayesRiskForSSELoss, EDL_DataLoss,
    EUC_RegularizationLoss, KL_Divergence_RegularizationLoss, EDL_RegularizationLoss,
    LinearAnnealingFactor, ExpAnnealingFactor,
)
from edl.run import run as edl_run
from edl.run import eval_run as edl_eval_run
from edl.grid_search_uncertainty_thresholds import stats_plot

DATASETS = [
    "mnist",
    "cifar10",
]

MODELS = [
    "lenet",
    "conv_net",
]

def _get_train_loader(args, batch_size, samples, train_loader_func) -> DataLoader:
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True
        }
        dataloader_kwargs.update(cuda_kwargs)
    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = train_loader_func(
        transform=transform,
        subset_first_n_classes=args.classes,
        n_samples=samples,
        kwargs=dataloader_kwargs,
    )
    return train_loader

def _get_test_loader(args, batch_size, samples, test_loader_func) -> DataLoader:
    dataloader_kwargs = {'batch_size': batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True
        }
        dataloader_kwargs.update(cuda_kwargs)
    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = test_loader_func(
        transform=transform,
        subset_first_n_classes=args.classes,
        n_samples=samples,
        kwargs=dataloader_kwargs,
    )
    return test_loader

def _get_train_and_test_loader(args, train_loader_func, test_loader_func) -> Tuple[DataLoader, DataLoader]:
    train_loader = _get_train_loader(args, args.batch_size, args.samples,train_loader_func)
    test_loader = _get_test_loader(args, args.test_batch_size, args.test_samples, test_loader_func)
    return train_loader, test_loader


def _get_uncertainty_loss(args) -> EDL_DataLoss:
    loss_name = args.uncertainty_loss.lower()
    if loss_name == "ml":
        return Type2MaximumLikelihoodLoss()
    elif loss_name == "ce":
        return BayesRiskForCrossEntropyLoss()
    elif loss_name == "sse":
        return BayesRiskForSSELoss()


def _get_uncertainty_regularization(args) -> EDL_RegularizationLoss:
    regularization_name = args.uncertainty_regularization.lower()
    if regularization_name == "kl":
        annealing_factor = LinearAnnealingFactor()
        return KL_Divergence_RegularizationLoss(annealing_factor)
    elif regularization_name == "euc":
        annealing_factor = ExpAnnealingFactor()
        return EUC_RegularizationLoss(annealing_factor)


def _get_criterion(args) -> Loss:
    if args.uncertainty_loss:
        criterion = _get_uncertainty_loss(args)
        if args.uncertainty_regularization:
            criterion += _get_uncertainty_regularization(args)
    else:
        criterion = NLL_Loss()
    return criterion

def _load_state(args, model, optimizer) -> Module:
    checkpoint = torch.load(args.checkpoint)
    preceding_epochs = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.load_optimizer:
        optimizer.load_state_dict(checkpoint["model_state_dict"])
    return preceding_epochs, model, optimizer

def _load_model_state(args, model) -> Module:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _create_model(args, device) -> Module:
    model_args = []
    model_kwargs = {}

    if args.dataset.lower() == "mnist":
        from dataloader.mnist import CLASSES
        
        if args.model.lower() == "lenet":
            from models.lenet import LeNet_MNIST as ModelCLS
            model_kwargs = {
                "dropout": True
            }

        elif args.model.lower() == "conv_net":
            from models.conv_net import ConvNet_MNIST as ModelCLS

    elif args.dataset.lower() == "cifar10":
        from dataloader.cifar10 import CLASSES

        if args.model.lower() == "conv_net":
            from models.conv_net import ConvNet_CIFAR10 as ModelCLS

    try:
        CLASSES
    except NameError:
        raise ValueError(f"Dataset {args.dataset} not supported")

    try:
        ModelCLS
    except NameError:
        raise ValueError(f"Model {args.model} not supported for dataset {args.dataset}")
    
    num_of_classes = CLASSES if args.classes is None else args.classes
    model = ModelCLS(num_of_classes, *model_args, **model_kwargs).to(device)
    return model


def _get_dataloader_functions(args):
    if args.dataset.lower() == "mnist":
        from dataloader.mnist import get_train_loader, get_test_loader
    elif args.dataset.lower() == "cifar10":
        from dataloader.cifar10 import get_train_loader, get_test_loader
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    return get_train_loader, get_test_loader


def _get_dataloader(args) -> Tuple[Module, DataLoader, DataLoader]:
    get_train_loader, get_test_loader = _get_dataloader_functions(args)
    train_loader, test_loader = _get_train_and_test_loader(args, get_train_loader, get_test_loader)
    return train_loader, test_loader


def _general_setup(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    print(f"Use CUDA: {use_cuda}")
    print(f"Use MPS: {use_cuda}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device(use_cuda, use_mps)
    model = _create_model(args, device)

    return model, device


def _train_setup(args):
    model, device = _general_setup(args)

    train_loader, test_loader = _get_dataloader(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = _get_criterion(args)

    if args.checkpoint:
        preceding_epochs, model, optimizer = _load_state(args, model, optimizer)
    else:
        preceding_epochs = 0

    model_name = (
        f"{args.dataset}_{args.model}_{args.uncertainty_loss if args.uncertainty else 'nll'}"
        f"{f'_{args.uncertainty_regularization}' if args.uncertainty and args.uncertainty_regularization is not None else ''}"
    )

    return model, device, train_loader, test_loader, criterion, optimizer, scheduler, preceding_epochs, model_name


def _eval_setup(args):
    model, device = _general_setup(args)
    model = _load_model_state(args, model)

    _, test_loader_func = _get_dataloader_functions(args)
    test_loader = _get_test_loader(args, args.batch_size, args.samples, test_loader_func)
    return model, device, test_loader


def create_parser():
    parser = argparse.ArgumentParser(description='EDL Playground')
    
    subparsers = parser.add_subparsers(dest="subcommand", help="Desired action to perform", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('--dataset', choices=DATASETS, default="mnist",
                        help='Dataset to use')
    parent_parser.add_argument('--model', choices=MODELS, default="lenet",
                        help='Model to use')

    parent_parser.add_argument('--uncertainty', action='store_true',
                        help='Use uncertainty')
    parent_parser.add_argument('--no-uncertainty', dest='uncertainty', action='store_false',
                        help='Do not use uncertainty')
    parent_parser.set_defaults(uncertainty=True)

    parent_parser.add_argument('--classes', type=int, default=None, metavar='N',
                        help='Only use the first N classes')
    parent_parser.add_argument('--samples', type=int, default=None, metavar='N',
                        help='Only use N samples per class')
    parent_parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Batch size to use (default: 64)')

    parent_parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parent_parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parent_parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')

    parent_parser.add_argument('--checkpoint', type=str, default=None, metavar="PATH",
                        help="Path to model checkpoint file")

    parent_train_parser = argparse.ArgumentParser(add_help=False)

    parent_train_parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parent_train_parser.add_argument('--test-samples', type=int, default=None, metavar='N',
                        help='Only use N test samples per class')

    parent_train_parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parent_train_parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parent_train_parser.add_argument('--weight_decay', type=float, default=0.005, metavar='DECAY',
                        help='weight decay (default: 0.005)')
    parent_train_parser.add_argument('--step_size', type=int, default=7, metavar='M',
                        help='Period of learning rate decay (default: 7)')
    parent_train_parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')

    parent_train_parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    parent_train_parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parent_train_parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints",
                        help="Directory for saving model checkpoints")
    parent_train_parser.add_argument('--thresh-for-best-model-acc', action='store_true',
                        help='Use the test accuracy considering the uncertainty threshold for determining the best model')
    parent_train_parser.add_argument('--no-thresh-for-best-model-acc', dest='uncertainty', action='store_false',
                        help='Do not use the test accuracy considering the uncertainty threshold for determining the best model')
    parent_train_parser.set_defaults(thresh_for_best_model_acc=True)

    parent_train_parser.add_argument('--load-optimizer', action='store_true',
                        help='Also load optimizer state from checkpoint file')
    parent_train_parser.add_argument('--no-load-optimizer', dest='load_optimizer', action='store_false',
                        help='Do not load optimizer state from checkpoint file')
    parent_train_parser.set_defaults(load_optimizer=True)

    parent_train_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parent_train_parser.add_argument('--stats-plot', action='store_true',
                        help='Display stats plot')
    parent_train_parser.add_argument('--no-stats-plot', dest='stats_plot', action='store_false',
                        help='Do not display stats plot')
    parent_train_parser.set_defaults(stats_plot=True)

    parent_train_parser.add_argument('--uncertainty-loss', choices=list(EDL_DATALOSSES.keys()), default="sse",
                        help='Loss function to use when using uncertainty')
    parent_train_parser.add_argument('--uncertainty-regularization', choices=list(EDL_REGULARIZATIONS.keys()) + [None], default="euc",
                        help='Regularization to use when using uncertainty')

    parent_eval_parser = argparse.ArgumentParser(add_help=False)
    
    parent_eval_parser.add_argument('--uncertainty-thresh', type=float, default=0.9,
                        help='Uncertainty threshold above which the model is assumed to reject making predictions (predicts "I do not know")')

    train_parser = subparsers.add_parser("train", parents=[parent_parser, parent_train_parser, parent_eval_parser],
                        help="Train a model")

    eval_parser = subparsers.add_parser("eval", parents=[parent_parser, parent_eval_parser],
                        help="Evaluate a model")

    grid_search_uncertainty_thresh_parser = subparsers.add_parser("grid-search-uncertainty-thresh", parents=[parent_parser, parent_train_parser, parent_eval_parser],
                        help="Run grid search over uncertainty thresholds")

    grid_search_uncertainty_thresh_group = grid_search_uncertainty_thresh_parser.add_mutually_exclusive_group()
    grid_search_uncertainty_thresh_group.add_argument('--thresholds', metavar='N', type=float, nargs='*',
                        help='List of uncertainty threshold for grid search above which the model is assumed to reject making predictions (predicts "I do not know")')
    grid_search_uncertainty_thresh_group.add_argument('--range', nargs="*", metavar=('start', 'end', 'step'), type=float,
                        help='Range [start, end] of uncertainty threshold for grid search above which the model is assumed to reject making predictions (predicts "I do not know")')
    
    oos_parser = subparsers.add_parser("test-out-of-sample", parents=[parent_parser],
                        help="Test a model out of sample")
    oos_parser.add_argument('--output_prefix', metavar='PREFIX', type=str, default="", help='Prefix to use for each output file')

    return parser


def _train(args):
    model, device, train_loader, test_loader, criterion, optimizer, scheduler, preceding_epochs, model_name = _train_setup(args)

    if args.uncertainty:
        best_model, train_metrics, test_metrics = edl_run(args.epochs, scheduler, args.stats_plot, model, device, train_loader, test_loader, criterion, optimizer, args.uncertainty_thresh, args.log_interval, args.dry_run, args.checkpoint_dir, args.save_model, model_name, preceding_epochs, args.thresh_for_best_model_acc)
    else:
        best_model, train_loss, train_acc, test_loss, test_acc = run(args.epochs, scheduler, args.stats_plot, model, device, train_loader, test_loader, criterion, optimizer, args.log_interval, args.dry_run, args.checkpoint_dir, args.save_model, model_name, preceding_epochs)

    if args.uncertainty:
        return best_model, train_metrics, test_metrics
    else:
        return best_model, train_loss, train_acc, test_loss, test_acc


def _eval(args):
    model, device, test_loader = _eval_setup(args)

    if args.uncertainty:
        test_metrics = edl_eval_run(model, device, test_loader, args.uncertainty_thresh)
    else:
        criterion = NLL_Loss()
        test_acc, test_rejected_corrects = eval_run(model, device, test_loader, criterion)


def grid_search_uncertainty_thresh(args):
    if not args.uncertainty:
        raise argparse.ArgumentTypeError("subcommand grid-search-uncertainty-thresh requires argument uncertainty")
    elif not args.thresholds and not args.range:
        raise argparse.ArgumentTypeError("Both thresholds and range arguments are not specified for subcommand grid-search-uncertainty-thresh")
    elif args.range and len(args.range) != 3:
        raise argparse.ArgumentTypeError(f"3 elements must be passed to range argument for subcommand grid-search-uncertainty-thresh (passed: {args.range})")
    elif args.range:
        args.thresholds = torch.arange(args.range[0], args.range[1] + args.range[2], args.range[2]).tolist()

    train_loss = []
    train_acc = []
    train_rejected_corrects = []
    test_acc = []
    test_rejected_corrects = []

    for thresh in args.thresholds:
        print("Uncertainty threshold:", thresh)
        args.uncertainty_thresh = thresh
        best_model, train_loss_, train_acc_, train_rejected_corrects_, test_acc_, test_rejected_corrects_ = _train(args)
        train_loss.append(train_loss_[-1])
        train_acc.append(train_acc_[-1])
        train_rejected_corrects.append(train_rejected_corrects_[-1])
        test_acc.append(test_acc_[-1])
        test_rejected_corrects.append(test_rejected_corrects_[-1])

    fig, axs = stats_plot(args.thresholds, train_loss, train_acc, train_rejected_corrects, test_acc, test_rejected_corrects)
    plt.show()

    return train_loss, train_acc, train_rejected_corrects, test_acc, test_rejected_corrects

def test_out_of_sample(args):
    from torchvision.datasets import MNIST
    from test_out_of_sample.test import rotating_image_classification, test_single_image

    model, device = _general_setup(args)
    model = _load_model_state(args, model)
    model.eval()

    test_set = MNIST("./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    digit_one, _ = test_set[5]

    prefix = args.output_prefix + "_" if args.output_prefix else ""

    rotating_image_classification(
        model, digit_one, f"./test_out_of_sample/results/{prefix}rotate.jpg", uncertainty=args.uncertainty
    )

    test_single_image(model, "./test_out_of_sample/data/one.jpg", f"./test_out_of_sample/results/{prefix}one.jpg", uncertainty=args.uncertainty)
    test_single_image(model, "./test_out_of_sample/data/yoda.jpg", f"./test_out_of_sample/results/{prefix}yoda.jpg", uncertainty=args.uncertainty)


def main(args):
    if args.subcommand == 'train':
        _train(args)
    elif args.subcommand == 'eval':
        _eval(args)
    elif args.subcommand == 'grid-search-uncertainty-thresh':
        grid_search_uncertainty_thresh(args)
    elif args.subcommand == 'test-out-of-sample':
        test_out_of_sample(args)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)