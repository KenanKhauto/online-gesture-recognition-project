import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "7592047, Kenan Khauto"

class Bottleneck3D(nn.Module):
    """
    A bottleneck layer for 3D ResNet.
    
    Attributes:
        expansion (int): Factor to expand the number of channels.
        conv1 (nn.Conv3d): First convolutional layer.
        bn1 (nn.BatchNorm3d): Batch normalization after the first convolution.
        conv2 (nn.Conv3d): Second convolutional layer.
        bn2 (nn.BatchNorm3d): Batch normalization after the second convolution.
        conv3 (nn.Conv3d): Third convolutional layer.
        bn3 (nn.BatchNorm3d): Batch normalization after the third convolution.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential): Downsampling layers if stride is not 1.
        dropout (nn.Dropout): Dropout layer.
    
    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels for the conv2 layer.
        stride (int): Stride size for the conv2 layer.
        downsample (nn.Sequential, optional): Downsampling layer if stride is not 1.
        use_dropout (bool): Whether to use dropout.
        dropout_prob (float): Dropout probability.
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, use_dropout=False, dropout_prob=0.5):
        super(Bottleneck3D, self).__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_prob)

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass for the Bottleneck3D.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after passing through the bottleneck layer.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        return out

class ResNet3D(nn.Module):
    """
    A 3D ResNet model for video processing.
    
    Attributes:
        in_channels (int): Number of input channels for the first layer.
        conv1 (nn.Conv3d): Initial 3D convolution layer.
        bn1 (nn.BatchNorm3d): Batch normalization for the initial convolution.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool3d): Max pooling layer.
        layer1, layer2, layer3, layer4 (nn.Sequential): ResNet layers.
        avgpool (nn.AdaptiveAvgPool3d): Adaptive average pooling layer.
        fc (nn.Linear): Fully connected layer.
        dropout (nn.Dropout): Dropout layer.
    
    Args:
        block (nn.Module): Block type to use (Bottleneck3D).
        layers (list of int): Number of blocks in each of the 4 layers of the network.
        num_classes (int): Number of classes for classification.
        use_dropout (bool): Whether to use dropout.
        dropout_prob (float): Dropout probability.
    """

    def __init__(self, block, layers, num_classes=14, use_dropout=False, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_prob)

        # Initial 3D convolution
        self.conv1 = nn.Conv3d(3, self.in_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], use_dropout=use_dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_dropout=use_dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_dropout=use_dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_dropout=use_dropout)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1, use_dropout=False):
        """
        Creates a ResNet layer with the specified number of blocks.
        
        Args:
            block (nn.Module): Block type to use (Bottleneck3D).
            channels (int): Number of channels in the block.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride size for the first block of the layer.
            use_dropout (bool): Whether to use dropout in the blocks.
        
        Returns:
            nn.Sequential: The constructed ResNet layer.
        """
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample, use_dropout))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, use_dropout=use_dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the ResNet3D.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after passing through the network.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)

        return x

def get_resnet101_3d():
    model = ResNet3D(Bottleneck3D, [3, 4, 23, 3], use_dropout=True, dropout_prob=0.5)
    return model

if __name__ == "__main__":
    model = get_resnet101_3d()
    with torch.no_grad():
        inputs = torch.randn((5, 3, 142, 32, 32))
        outputs = model(inputs)

        print(outputs.shape)
