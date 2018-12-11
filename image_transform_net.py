"""
Image Transformation Network for style transfer.

Written by Zachary Ferguson
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class ResidualBlock(nn.Module):
    """Redisual network block for style transfer."""

    def __init__(self, nchannels):
        """Create a block of a residual network."""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.conv2 = nn.Conv2d(nchannels, nchannels, kernel_size=3)

        self.norm_conv1 = nn.InstanceNorm2d(nchannels, affine=True)
        self.norm_conv2 = nn.InstanceNorm2d(nchannels, affine=True)

        # Use a ReLU function as the nonlinear activation
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        """Forward the input through the block."""
        residual = x[:, :, 2:-2, 2:-2]
        out = self.nonlinearity(self.norm_conv1(self.conv1(x)))
        out = self.norm_conv2(self.conv2(out))
        return out + residual


class UpsampleConv2d(nn.Module):
    """
    Avoid checkerboard patterns by upsampling the image and convolving.

    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 upsample):
        """Set parameters for upsampling."""
        super(UpsampleConv2d, self).__init__()
        self.upsample = upsample
        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """
        Upsample then convolve the image.

        "We’ve had our best results with nearest-neighbor interpolation, and
        had difficulty making bilinear resize work. This may simply mean that,
        for our models, the nearest-neighbor happened to work well with
        hyper-parameters optimized for deconvolution. It might also point at
        trickier issues with naively using bilinear interpolation, where it
        resists high-frequency image features too strongly. We don’t
        necessarily think that either approach is the final solution to
        upsampling, but they do fix the checkerboard artifacts."
        (https://distill.pub/2016/deconv-checkerboard/)
        """
        # Use nearest neighbor interpolation because of the above
        x = F.interpolate(x, mode="nearest", scale_factor=self.upsample)
        return self.conv(self.padding(x))


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
        self.norm_conv1 = nn.InstanceNorm2d(32, affine=True)
        self.norm_conv2 = nn.InstanceNorm2d(64, affine=True)
        self.norm_conv3 = nn.InstanceNorm2d(128, affine=True)

        # Residual Blocks
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.res_block3 = ResidualBlock(128)
        self.res_block4 = ResidualBlock(128)
        self.res_block5 = ResidualBlock(128)

        # Upsample convolution networks
        self.upsample_conv1 = UpsampleConv2d(
            128, 64, kernel_size=3, stride=1, padding=1, upsample=2)
        self.upsample_conv2 = UpsampleConv2d(
            64, 32, kernel_size=3, stride=1, padding=1, upsample=2)
        self.upsample_conv3 = UpsampleConv2d(
            32, 3, kernel_size=9, stride=1, padding=4, upsample=1)
        self.norm_upsample_conv1 = nn.InstanceNorm2d(64, affine=True)
        self.norm_upsample_conv2 = nn.InstanceNorm2d(32, affine=True)
        self.norm_upsample_conv3 = nn.InstanceNorm2d(3, affine=True)

        # Use a ReLU function as the nonlinear activation
        self.nonlinearity = nn.ReLU()
        # Normalize the output to a range [0, 255]
        self.tanh = nn.Tanh()
        self.output_nonlineaity = lambda x: (self.tanh(x) + 1) / 2 * 255

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
        x = self.nonlinearity(self.norm_upsample_conv1(self.upsample_conv1(x)))
        x = self.nonlinearity(self.norm_upsample_conv2(self.upsample_conv2(x)))
        x = self.norm_upsample_conv3(self.upsample_conv3(x))
        return self.output_nonlineaity(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    print(numpy.max(x.numpy()))
    net = ImageTransformNet()
    x = net(x)
    print(x.shape)
    print(numpy.max(x.detach().numpy()))
