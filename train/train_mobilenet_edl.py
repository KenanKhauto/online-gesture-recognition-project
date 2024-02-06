import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.transforms import transforms
from dataset.loader import GestureDataset
from solver.edl_solver import Solver
from models.mobilenet import get_mobilenet

from edl_playground.edl.losses import (
    LinearAnnealingFactor, ExpAnnealingFactor,
    Type2MaximumLikelihoodLoss, BayesRiskForCrossEntropyLoss, BayesRiskForSSELoss,
    KL_Divergence_RegularizationLoss, EUC_RegularizationLoss,
)


def main(path_frames, path_annotations_train, path_annotations_test, path_to_save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=60)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=60)

    val_fraction = 0.3
    val_samples = int(val_fraction * (len(ds_train) + len(ds_test)))
    ds_train, ds_val = random_split(ds_train, [len(ds_train) - val_samples, val_samples])

    num_classes = 14

    model = get_mobilenet(num_classes, softmax=False)

    annealing_steps = 10
    annealing_factor = LinearAnnealingFactor(annealing_steps)
    criterion = BayesRiskForSSELoss() + KL_Divergence_RegularizationLoss(annealing_factor)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    uncertainty_thresh = 0.5

    solver = Solver(
        model, 
        ds_train, 
        ds_val, 
        criterion, 
        optimizer, 
        scheduler, 
        device,
        uncertainty_thresh,
        batch_size=64,
        cnn_trans=False,
        distr=False,
        detector=False,
        save_every=5,
        path_to_save=path_to_save,
        num_classes=num_classes,
        best_model_metric="Val_AccU",
        uncertainty_distribution=True,
    )
    results = solver.train(40)

    solver.save(path_to_save)

    with open("mobilenet_edl_train_results.json", "w") as f:
        json.dump(results, f)

    results = solver.test(ds_test)

    with open("mobilenet_edl_test_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run.py <path_frames> <path_annotations_train> <path_annotations_test> <path_to_save>")
        sys.exit(1)
    path_frames = sys.argv[1]
    path_annotations_train = sys.argv[2]
    path_annotations_test = sys.argv[3]
    path_to_save = sys.argv[4]
    main(path_frames, path_annotations_train, path_annotations_test, path_to_save)