
import json
import sys
#from solver import Solver
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from dataset.loader import GestureDataset
from solver.edl_solver import Solver
from models.cnn_transformer import get_resnet_transformer

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

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=224)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=224)

    model = get_resnet_transformer(64, 16, 64, 8)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    annealing_steps = 10
    annealing_factor = LinearAnnealingFactor(annealing_steps)
    criterion = BayesRiskForSSELoss() + KL_Divergence_RegularizationLoss(annealing_factor)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    uncertainty_thresh = 0.5

    solver = Solver(model, ds_train, ds_test, criterion, optimizer, scheduler, device, uncertainty_thresh, cnn_trans=True, consider_uncertainty_for_best_model_val_accuracy=True)
    results = solver.train(20)

    solver.save(path_to_save)

    with open("results.json", "w") as f:
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