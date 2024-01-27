
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from dataset.loader import GestureDataset
from solver.solver import Solver
from webcam_capturing.ResNetL import resnetl10
import os
import torch.distributed as dist

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

    ds_train = GestureDataset(path_frames, 
                              path_annotations_train, 
                              transform, 
                              sample_duration=60, 
                              label_all_gestures=True)
    
    ds_test = GestureDataset(path_frames, 
                             path_annotations_test, 
                             transform, 
                             sample_duration=60, 
                             label_all_gestures=True)

    model = resnetl10(num_classes = 2, sample_size = 128, sample_duration=60)

    n_gesture = 3117
    n_no_gesture = 4039 - 3117
    weight = n_gesture / n_no_gesture

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    solver = Solver(model, 
                    ds_train, 
                    ds_test, 
                    criterion, 
                    optimizer, 
                    scheduler, 
                    device, 
                    batch_size=32, 
                    cnn_trans=False, 
                    detector=True,
                    save_every=5,
                    path_to_save=path_to_save)
    results = solver.train(60)

    if distr:
        if rank == 0:  # Save model and results in the main process
            solver.save(path_to_save)
            with open("detector.json", "w") as f:
                json.dump(results, f)

        cleanup()
    else:
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
        main(None, 
             path_frames, 
             path_annotations_train, 
             path_annotations_test, 
             path_to_save, distr, 
             world_size=None)
   