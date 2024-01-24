from fastapi import FastAPI 
import uvicorn

from webcam_capturing.WebcamCapture import WebcamCapture
from  webcam_capturing.ResNetL import  resnetl10 

from time import sleep
import torch 


app = FastAPI()
detector = resnetl10(num_classes = 2, sample_size = 32, sample_duration=60)
detector.load_state_dict(torch.load("resnetl_detector.pth", map_location=torch.device('cpu')))

@app.get('/test')
async def test():
    return {"message" : 'hello'}
    

@app.get('/webcam')
async def run_webcam_get_tensors():
    webcam_capture = WebcamCapture()
    webcam_capture.check_manually_webcam_frame_rate()
    frames_tensor = webcam_capture.start_capture(save_test_frames=False)
    return {'Tensor shape' : frames_tensor.shape }

@app.get('/run')
async def run_model_on_webcam():
    # Disable gradient computation for efficiency and to avoid unnecessary memory usage
    with torch.no_grad():
        x = torch.rand((60, 3, 60, 32, 32))
        pred = detector(x)
        return {'test' : pred}

    

if __name__ == '__main__':
    uvicorn.run('api:app',host='127.0.0.1',port=8000,reload=True)