
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
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, path_frames, path_annotations_train, path_annotations_test, path_to_save):

    setup(rank, world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=142)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=142)

    model = get_resnet_transformer(64, 16, 64, 4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    solver = Solver(model, ds_train, ds_test, criterion, optimizer, scheduler, device, rank, world_size, cnn_trans=True)
    results = solver.train(20)

    if rank == 0:  # Save model and results in the main process
        solver.save(path_to_save)
        with open("3d_resnet_results.json", "w") as f:
            json.dump(results, f)

    cleanup()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run.py <path_frames> <path_annotations_train> <path_annotations_test> <path_to_save>")
        sys.exit(1)
    path_frames = sys.argv[1]
    path_annotations_train = sys.argv[2]
    path_annotations_test = sys.argv[3]
    path_to_save = sys.argv[4]
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, 
                                args=(world_size, 
                                      path_frames, 
                                      path_annotations_train, 
                                      path_annotations_test, 
                                      path_to_save), 
                                nprocs=world_size, 
                                join=True)