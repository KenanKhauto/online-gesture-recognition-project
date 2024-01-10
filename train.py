
import json
import sys
#from solver import Solver
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from dataset.loader import GestureDataset
from solver.solver import Solver
from models.cnn_transformer import get_resnet_transformer


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    solver = Solver(model, ds_train, ds_test, criterion, optimizer, scheduler, device, cnn_trans=True)
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