import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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