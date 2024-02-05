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
from models.LSTM import LSTMGestureClassifier
import mediapipe as mp 


def main(path_frames, path_annotations_train, path_annotations_test, path_to_save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    ds_train = GestureDataset(path_frames, path_annotations_train, transform, sample_duration=60, label_all_gestures=True)
    ds_test = GestureDataset(path_frames, path_annotations_test, transform, sample_duration=60, label_all_gestures=True)

    model = LSTMGestureClassifier(63, 128, 2, 2)
    weight_for_no_gesture = 0.8 
    weight_for_gesture = 0.2     
    weights = torch.tensor([weight_for_gesture, weight_for_no_gesture]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    # mp_drawing = mp.solutions.drawing_utils

    solver = Solver(model, 
                    ds_train, 
                    ds_test, 
                    criterion, 
                    optimizer, 
                    None, 
                    device, 
                    batch_size=5,
                    cnn_trans=False,
                    detector=False,
                    save_every=5,
                    path_to_save=path_to_save,
                    use_lstm=True,
                    hand_marks_detector=hands,
                    num_classes=2)
    
    results = solver.train(25)

    solver.save(path_to_save)

    with open("lstm_detector_results.json", "w") as f:
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