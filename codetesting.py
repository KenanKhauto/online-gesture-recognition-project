from fastapi import FastAPI 
import uvicorn

from webcam_capturing.WebcamCapture import WebcamCapture
from  webcam_capturing.ResNetL import  resnetl10 

from time import sleep
import torch 

detector = resnetl10(num_classes = 2, sample_size = 32, sample_duration=60)
detector.load_state_dict(torch.load("resnetl_detector.pth", map_location=torch.device('cpu')))

def run_model_on_cam():
    
    
    pass

    # Disable gradient computation for efficiency and to avoid unnecessary memory usage
    # with torch.no_grad():
    #     pred = detector(frame)
    #     print(pred)

if __name__ == '__main__':
    run_model_on_cam()    