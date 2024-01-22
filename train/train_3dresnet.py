import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from dataset.loader import GestureDataset
from solver.solver import Solver
from models.resnet101_3d_cnn import get_resnet101_3d


def main(path_frames, path_annotations_train, path_annotations_test, path_to_save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=60)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=60)

    model = get_resnet101_3d()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    solver = Solver(model, 
                    ds_train, 
                    ds_test, 
                    criterion, 
                    optimizer, 
                    None, 
                    device, 
                    batch_size=2,
                    cnn_trans=False,
                    detector=False,
                    save_every=5,
                    path_to_save=path_to_save)
    results = solver.train(40)

    solver.save(path_to_save)

    with open("3d_resnet_results.json", "w") as f:
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