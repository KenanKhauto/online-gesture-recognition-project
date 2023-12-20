
import torch
import torch.nn as nn

__author__ = "7592047, Kenan Khauto"

class Bottleneck(nn.Module):
    """
    A bottleneck block for ResNet.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with kernel size 1.
        bn1 (nn.BatchNorm2d): Batch normalization after conv1.
        conv2 (nn.Conv2d): Second convolutional layer with kernel size 3.
        bn2 (nn.BatchNorm2d): Batch normalization after conv2.
        conv3 (nn.Conv2d): Third convolutional layer with kernel size 1.
        bn3 (nn.BatchNorm2d): Batch normalization after conv3.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential): Downsampling layer if stride is not 1.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolutional layers.
        stride (int): Stride of the convolution.
        downsample (nn.Module, optional): Downsampling layer for the residual connection.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor of the block.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet model with Bottleneck blocks.

    Attributes:
        in_channels (int): Internal tracker for in_channels between layers.
        conv1 (nn.Conv2d): Initial convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the initial convolution.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool2d): Max pooling layer following the initial convolution.
        layer1 (nn.Sequential): First set of layers in ResNet.
        layer2 (nn.Sequential): Second set of layers in ResNet.
        layer3 (nn.Sequential): Third set of layers in ResNet.
        layer4 (nn.Sequential): Fourth set of layers in ResNet.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.

    Args:
        block (nn.Module): Type of block to use in the model (Bottleneck).
        layers (list of int): Number of blocks in each of the four layers of the network.
        num_classes (int): Number of classes for the final classification layer.
    """
    def __init__(self, block, layers, num_classes=14):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a layer of blocks for ResNet.

        Args:
            block (nn.Module): Type of block to use in the layer.
            out_channels (int): Number of channels in the output from each block.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride to use in the first block of the layer.

        Returns:
            nn.Sequential: The constructed layer of blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the ResNet model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def get_resnet101():
    """returns the resnet101 model"""
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

if __name__ == "__main__":
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    with torch.no_grad():
        inputs = torch.randn((5, 3, 32, 32))
        outputs = model(inputs)
        print(outputs.shape)
