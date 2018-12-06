"""
Compute the style and content loss using a VGG-16 model trained on ImageNet.

Uses perceptual loss to compute the style and content loss.
"""
import torch
import torchvision

from collections import namedtuple

LossOutput = namedtuple(
    "LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])


class LossNet(torch.nn.Module):
    """
    Compute the losses for neural style transfer.

    https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        """Initialize the layers to use for loss computation."""
        super(LossNet, self).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2",
                                   "15": "relu3_3", "22": "relu4_3"}
        for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass through the vgg to compute the perceptual loss."""
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
