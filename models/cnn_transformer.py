import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from resnet101_2d_cnn import get_resnet101

__author__ = "7592047, Kenan Khauto"

class CNNTransformerVariable(nn.Module):
    """
    A model combining a CNN and a Transformer for gesture recognition with variable-length input.

    Attributes:
        cnn (nn.Module): Convolutional neural network for feature extraction.
        transformer_encoder (TransformerEncoder): Transformer encoder for processing sequences.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        fc (nn.Linear): Fully connected layer for classification.

    Args:
        cnn_model (nn.Module): Predefined CNN model for feature extraction.
        ntoken (int): Size of the feature vectors (output of CNN).
        nhead (int): Number of heads in the multiheadattention model.
        nhid (int): Dimension of the feedforward network model in nn.TransformerEncoder.
        nlayers (int): Number of nn.TransformerEncoderLayer in nn.TransformerEncoder.
        num_classes (int): Number of classes for the final classification layer.
        dropout (float): Dropout value.
    """
    def __init__(self, cnn_model, ntoken, nhead, nhid, nlayers, num_classes, dropout=0.5):
        super(CNNTransformerVariable, self).__init__()
        self.cnn = cnn_model
        self.pos_encoder = PositionalEncoding(ntoken, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=ntoken, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fc = nn.Linear(ntoken, num_classes)

    def forward(self, src):
        """
        Forward pass of the CNNTransformerVariable model.

        Args:
            src (Tensor): Input tensor with shape (batch_size, max_sequence_length, C, H, W).

        Returns:
            Tensor: Output tensor after classification.
        """
        batch_size, max_sequence_length, C, H, W = src.shape

        src_mask = self._create_src_mask(src) 

        src = src.view(batch_size * max_sequence_length, C, H, W)
        src = self.cnn(src)
        src = src.view(batch_size, max_sequence_length, -1)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.fc(output.mean(dim=1))
        return output
    
    def _create_src_mask(self, batch_sequences, pad_token=0):
        """
        Create a source mask for the Transformer given a batch of padded sequences.
        
        Args:
            batch_sequences (Tensor): Tensor of shape (batch_size, max_seq_length, C, H, W)
                                    containing the batch of padded sequences.
            pad_token (int): The padding token used in the sequences (default: 0).

        Returns:
            Tensor: A boolean mask tensor of shape (batch_size, max_seq_length) where `True` 
                    values correspond to padded positions.
        """
        # Check if a frame is entirely padded (assuming padding is done with zeros)
        mask = batch_sequences[:, :, 0, 0, 0] == pad_token
        return mask.transpose(0, 1)  # Transpose to shape [seq_length, batch_size]


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (Tensor): Positional encoding tensor.

    Args:
        d_model (int): The dimension of the embeddings (feature vectors).
        dropout (float): Dropout value.
    """
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with positional encodings added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ModifiedResNet(nn.Module):
    """
    modifies the original resnet101 architecture to combine with 
    a transformer.

    Attributes:
        features (nn.Sequuential): contains all the layers of the original without the last layer
        fc (nn.Linear): the new modified fully connected layer

    Paramters:
        original_resnet (nn.Module): the original model that should be modified
        feature_size (int): the size of the vector for each frame, this should match ntoken from the transformer
    
    """
    def __init__(self, original_resnet, feature_size):
        super(ModifiedResNet, self).__init__()
        # Everything except the final layer
        self.features = nn.Sequential(*list(original_resnet.children())[:-1])
        self.fc = nn.Linear(original_resnet.fc.in_features, feature_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc(x)
        return x
    

def get_resnet_transformer(feature_size, nhead, nhid, nlayers, num_classes = 14, dropout_prob = 0.5):
    """
    returns the model that combines a transformer and a 2D resnet101

    Parameters:
        ntoken (int): Size of the feature vectors (output of CNN).
        nhead (int): Number of heads in the multiheadattention model.
        nhid (int): Dimension of the feedforward network model in nn.TransformerEncoder.
        nlayers (int): Number of nn.TransformerEncoderLayer in nn.TransformerEncoder.
        num_classes (int): Number of classes for the final classification layer.
        dropout (float): Dropout value.

    Returns:
        model (nn.module)
    """
    cnn = get_resnet101()
    mod_cnn = ModifiedResNet(cnn, feature_size)
    model = CNNTransformerVariable(mod_cnn, feature_size, nhead, nhid, nlayers, num_classes, dropout_prob)
    return model


if __name__ == "__main__":

    inputs = torch.randn((5, 3, 32, 32))
    cnn = get_resnet101()
    mod_cnn = ModifiedResNet(cnn, 64)
    # outputs = mod_cnn(inputs)
    # print(outputs) 
    # Modify the test data to simulate variable lengths with padding

    batch_size = 32
    num_frames = 142  # Max number of frames
    height, width = 32, 32
    channels = 3
    num_classes = 14  # Assuming 14 classes for classification

    # Simulate variable-length sequences
    # For simplicity, let's assume each sequence's length is randomly chosen
    lengths = torch.randint(50, num_frames, (batch_size,))  # Random lengths between 50 and 142
    test_data = torch.zeros(batch_size, num_frames, channels, height, width)  # Initialize with zeros (padded)

    for i in range(batch_size):
        seq_length = lengths[i]
        test_data[i, :seq_length] = torch.randn(seq_length, channels, height, width)  # Fill with random data


    combo = CNNTransformerVariable(mod_cnn, 64, 16, 32, 6, 14)

    with torch.no_grad():
        outputs = combo(test_data)
        print(outputs.shape)

    from utils import count_trainable_parameters
    print(count_trainable_parameters(combo))