"""
Compute the style and content loss using a VGG-16 model trained on ImageNet.

Uses perceptual loss to compute the style and content loss.
"""
import torch
import torchvision
from collections import namedtuple

LossOutput = namedtuple(
    "LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])


def gram_matrix(x):
    """Create the gram matrix of x."""
    b, c, h, w = x.shape
    phi = x.view(b, c, h * w)
    return phi.bmm(phi.transpose(1, 2)) / (c * h * w)


class PerceptualLossNet(torch.nn.Module):
    """
    Compute the losses for neural style transfer.

    Pretrained VGG network to return the relu values.
    https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self, content_weight, style_weights, regularization_weight):
        """Initialize the layers to use for loss computation."""
        super(PerceptualLossNet, self).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2",
                                   "15": "relu3_3", "22": "relu4_3"}
        for param in self.parameters():
                param.requires_grad = False
        self.ys_grams = None
        self.content_weight = content_weight
        self.style_weights = style_weights
        self.regularization_weight = regularization_weight
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x):
        """Forward pass through the vgg to compute the perceptual loss."""
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

    @staticmethod
    def normalize_batch(batch):
        """Normalize using imagenet mean and std."""
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch / 255.0
        return (batch - mean) / std

    def compute_perceptual_loss(self, y, yc, ys):
        """Compute the perceptual loss."""
        # Precompute the gram matrices of the style features
        if self.ys_grams is None:
            ys_features = self(self.normalize_batch(ys))
            self.ys_grams = [gram_matrix(feature) for feature in ys_features]

        # Compute the Loss Network features of the content and stylized
        # content.
        yc = self.normalize_batch(yc)
        yc_features = self(yc)
        y = self.normalize_batch(y)
        y_features = self(y)

        # Feature loss is the mean squared error of the content and
        # stylized content
        content_loss = self.content_weight * self.mse_loss(
            y_features.relu2_2, yc_features.relu2_2)

        # Style loss id the Frobenius norm of the gram matrices
        style_loss = 0.0
        for y_feature, ys_gram, style_weight in zip(
                y_features, self.ys_grams, self.style_weights):
            style_loss += style_weight * self.mse_loss(
                gram_matrix(y_feature), ys_gram[:yc.shape[0]])

        # Compute the regularized total variation of the stylized image
        total_variation = self.regularization_weight * (
            torch.sum(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])) +
            torch.sum(torch.abs(y[:, :, 1:, :] - y[:, :, -1:, :])))

        # The total loss is a weighted sum of the loss values
        # return content_loss + style_loss + total_variation
        return content_loss + style_loss
