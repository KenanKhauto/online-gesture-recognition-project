
import json
import sys
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from dataset.loader import GestureDataset
from solver.edl_solver import Solver
from models.cnn_transformer import get_resnet_transformer
import torch.distributed as dist

from edl_playground.edl.losses import (
    LinearAnnealingFactor, ExpAnnealingFactor,
    Type2MaximumLikelihoodLoss, BayesRiskForCrossEntropyLoss, BayesRiskForSSELoss,
    KL_Divergence_RegularizationLoss, EUC_RegularizationLoss,
)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, path_frames, path_annotations_train, path_annotations_test, path_to_save, distr, world_size):
    if distr:
        setup(rank, world_size)
        device = rank # torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=60)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=60)

    model = get_resnet_transformer(64, 16, 64, 4)

    annealing_steps = 10
    annealing_factor = LinearAnnealingFactor(annealing_steps)
    criterion = BayesRiskForSSELoss() + KL_Divergence_RegularizationLoss(annealing_factor)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    uncertainty_thresh = 0.5
    batch_size = 16

    solver = Solver(
        model, 
        ds_train, 
        ds_test,
        criterion,
        optimizer,
        scheduler,
        device,
        uncertainty_thresh,
        world_size,
        batch_size,
        cnn_trans=True,
        distr=distr,
        consider_uncertainty_for_best_model_val_accuracy=True,
    )
    results = solver.train(20)

    if distr:
        if rank == 0:  # Save model and results in the main process
            solver.save(path_to_save)
            with open("cnn_trans_results.json", "w") as f:
                json.dump(results, f)

        cleanup()
    else:
        solver.save(path_to_save)
        with open("cnn_trans_results.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run.py <path_frames> <path_annotations_train> <path_annotations_test> <path_to_save>")
        sys.exit(1)
    path_frames = sys.argv[1]
    path_annotations_train = sys.argv[2]
    path_annotations_test = sys.argv[3]
    path_to_save = sys.argv[4]
    distr = True

    if distr:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(main, 
                                    args=(path_frames, 
                                        path_annotations_train, 
                                        path_annotations_test, 
                                        path_to_save,
                                        distr,
                                        world_size), 
                                    nprocs=world_size, 
                                    join=True)
    else:
        main(None, path_frames, path_annotations_train, path_annotations_test, path_to_save, distr, world_size=None)