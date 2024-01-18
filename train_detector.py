
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from webcam_capturing.DetectorDatasetLoader import GestureDataset
from solver.solver import Solver
from webcam_capturing.ResNetL import resnetl10


def main(path_frames, path_annotations_train, path_annotations_test, path_to_save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=60)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=60)

    model = resnetl10(num_classes = 2, sample_size = 32, sample_duration=60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    solver = Solver(model, ds_train, ds_test, criterion, optimizer, scheduler, device, batch_size=32, cnn_trans=False)
    results = solver.train(20)

    solver.save(path_to_save)

    with open("detector.json", "w") as f:
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