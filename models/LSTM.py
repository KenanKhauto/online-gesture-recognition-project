import torch
import torch.nn as nn

class LSTMGestureClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTMGestureClassifier, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the fully connected (linear) layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x)
        
        # Get the output from the last time step (sequence classification)
        out = out[:, -1, :]
        
        # Forward pass through the linear layer for classification
        out = self.fc(out)
        return out
    


if __name__ == "__main__":
    model = LSTMGestureClassifier(3*21, 128, 2, 2)

    x = torch.rand((32, 60, 3*21))
    print(x.shape)
    out = model(x)

    print(out.shape)