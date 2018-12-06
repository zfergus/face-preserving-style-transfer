"""
Image Transformation Network for style transfer.

Written by Zachary Ferguson
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Redisual network block for style transfer."""

    def __init__(self):
        """Create a block of a residual network."""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)

        self.norm_conv1 = nn.InstanceNorm2d(128)
        self.norm_conv2 = nn.InstanceNorm2d(128)

        # Use a ReLU function as the nonlinear activation
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        """Forward the input through the block."""
        residual = x[:, :, 2:-2, 2:-2]
        out = self.nonlinearity(self.norm_conv1(self.conv1(x)))
        out = self.norm_conv2(self.conv2(out))
        out += residual
        return out


class ImageTransformNet(nn.Module):
    """
    Image Transformation Network for style transfer.

    See here for the architecture:
    https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
    """

    def __init__(self):
        """Construct the network to train later."""
        super(ImageTransformNet, self).__init__()

        # Spatial reflection padding so the input and output network have the
        # same size
        self.reflection_padding = nn.ReflectionPad2d(40)

        # Downsample convolution networks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Batch normalization for each convolution layer. This helps
        # normalize the data fed into the nonlinearity.
        self.norm_conv1 = nn.InstanceNorm2d(32, affine=True)
        self.norm_conv2 = nn.InstanceNorm2d(64, affine=True)
        self.norm_conv3 = nn.InstanceNorm2d(128, affine=True)

        # Residual Blocks
        self.res_block1 = ResidualBlock()
        self.res_block2 = ResidualBlock()
        self.res_block3 = ResidualBlock()
        self.res_block4 = ResidualBlock()
        self.res_block5 = ResidualBlock()

        # Upsample convolution networks
        self.conv_transpose1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(
            32, 3, kernel_size=9, stride=1, padding=4, output_padding=0)
        # Batch normalization for each convolution layer. This helps
        # normalize the data fed into the nonlinearity.
        self.norm_conv_transpose1 = nn.InstanceNorm2d(64, affine=True)
        self.norm_conv_transpose2 = nn.InstanceNorm2d(32, affine=True)
        self.norm_conv_transpose3 = nn.InstanceNorm2d(3, affine=True)

        # Use a ReLU function as the nonlinear activation
        self.nonlinearity = nn.ReLU()
        # Normalize the output to a range [0, 1]
        self.tanh = nn.Tanh()
        self.output_nonlineaity = lambda x: (self.tanh(x) + 1) / 2

    def forward(self, x):
        """Feed the data throught the network."""
        x = self.reflection_padding(x)
        x = self.nonlinearity(self.norm_conv1(self.conv1(x)))
        x = self.nonlinearity(self.norm_conv2(self.conv2(x)))
        x = self.nonlinearity(self.norm_conv3(self.conv3(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.nonlinearity(
            self.norm_conv_transpose1(self.conv_transpose1(x)))
        x = self.nonlinearity(
            self.norm_conv_transpose2(self.conv_transpose2(x)))
        x = self.output_nonlineaity(
            self.norm_conv_transpose3(self.conv_transpose3(x)))
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    net = ImageTransformNet()
    net(x)
    print(x.shape)
