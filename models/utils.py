import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_training(history):
    """
    Show loss and accuracies during training.

    Parameters:
        - history (dict):
            - loss (list[float]): Training losses.
            - train_acc (list[float]): Training accuracies.
            - val_acc (list[float]): Validation accuracies.

    """
    fig, (lhs, rhs) = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle('Training')

    # Set subplot titles.
    lhs.set_title('Loss')
    rhs.set_title('mAP')

    # Set subplot axis labels.
    lhs.set_xlabel('epoch'), lhs.set_ylabel('loss')
    rhs.set_xlabel('epoch'), rhs.set_ylabel('mAP')

    # Plot loss and accuracies.
    lhs.plot(history['loss'])
    rhs.plot(history['train_accuracy'], label='train')
    rhs.plot(history['val_accuracy'], label='val')
    rhs.legend()

    plt.show()



def process_video_for_landmarks(video_frames, device):
    try:
        # Check if the input is a PyTorch tensor
        if not isinstance(video_frames, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
        
        # Check if the tensor has the correct dimensions (sequence, channels, height, width)
        if len(video_frames.shape) != 4:
            raise ValueError("Input tensor must have shape (sequence, channels, height, width)")
        
        # Check if the tensor data type is float32
        if video_frames.dtype != torch.float32:
            raise ValueError("Input tensor must have data type float32")
        
        # Check if the tensor values are within the [0, 1] range
        if (video_frames < 0).any() or (video_frames > 1).any():
            raise ValueError("Tensor values must be in the range [0, 1]")
        
        video_frames = video_frames.permute(0, 2, 3, 1).cpu()
        
        # Convert the tensor to a NumPy array and scale the values to uint8 [0, 255] range
        processed_video = (video_frames.numpy() * 255).astype(np.uint8)
        processed_video = np.ascontiguousarray(processed_video, dtype=np.uint8)
        return processed_video
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None


def process_batch_for_landmarks(batch, device):
    batch_size = batch.shape[0]

    # out shape: shape (batch, sequence, height, width, channels)
    out = np.zeros((batch.shape[0], batch.shape[1], batch.shape[3], batch.shape[4], batch.shape[2]), np.uint8)
    for vidx in range(batch_size):
        vid = process_video_for_landmarks(batch[vidx], device)
        out[vidx] = vid

    return out


def process_image_for_landmarks(image):

    # Scale the image values to uint8 [0, 255] range
    image = (image * 255).astype(np.uint8)
    
    # Ensure the image is contiguous in memory
    image = np.ascontiguousarray(image, dtype=np.uint8)
    
    return image



def extract_landmarks_from_batch(batch_videos, hands, device):
    batch_size = batch_videos.shape[0]  # Number of videos in the batch
    num_frames = batch_videos.shape[1]  # Number of frames per video

    # Initialize an empty list to store landmarks for the entire batch
    batch_landmarks = []
    for video_idx in range(batch_size):
        video = batch_videos[video_idx]

        # Pre-allocate array for landmarks for this video
        video_landmarks = []
    
       
        for frame_idx in range(num_frames):
            frame = video[frame_idx]
            
            results = hands.process(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                    video_landmarks.append(landmarks)
                    
            else:
                video_landmarks.append(np.zeros((21, 3)))

        batch_landmarks.append(video_landmarks)

    return np.array(batch_landmarks).reshape(batch_size, num_frames, -1)

