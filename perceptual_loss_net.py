"""
Compute the style and content loss using a VGG-16 model trained on ImageNet.

Uses perceptual loss to compute the style and content loss.
"""
import pathlib
import numpy
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from collections import namedtuple

import mtcnn.detector
import openface.net

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
        # total_variation = self.regularization_weight * (
        #     torch.sum(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])) +
        #     torch.sum(torch.abs(y[:, :, 1:, :] - y[:, :, -1:, :])))

        # The total loss is a weighted sum of the loss values
        # return content_loss + style_loss + total_variation
        return content_loss + style_loss


class FacePerceptualLossNet(PerceptualLossNet):
    """Account for loss in facial recognition."""

    def __init__(self, content_weight, style_weights, regularization_weight):
        """Initialize a face detection model on top of the perceptual loss."""
        super(FacePerceptualLossNet, self).__init__(
            content_weight, style_weights, regularization_weight)
        self.face_recog_model = openface.net.model
        model_file = (pathlib.Path(__file__).resolve().parent /
                      "OpenFace-PyTorch" / "net.pth")
        self.face_recog_model.load_state_dict(torch.load(str(model_file)))
        self.face_recog_model.eval()
        self.n_faces_seen = 0

    def compute_perceptual_loss(self, y, yc, ys):
        """Compute the perceptual loss including the face loss."""
        loss = super(FacePerceptualLossNet, self).compute_perceptual_loss(
            y, yc, ys)

        facial_loss = 0.0
        for i, image in enumerate(yc):
            image_array = image.cpu().numpy().clip(0, 255)
            pil_image = Image.fromarray(
                image_array.transpose(1, 2, 0).astype("uint8"))
            bounding_boxes, landmarks = mtcnn.detector.detect_faces(pil_image)
            for face_bb in bounding_boxes:
                if face_bb[-1] > 0.9:
                    # Face found
                    self.n_faces_seen += 1
                    b = face_bb[:-1].round().astype("int")
                    b[::2] = numpy.clip(b[::2], 0, yc.shape[2])
                    b[1::2] = numpy.clip(b[1::2], 0, yc.shape[3])

                    yc_face = yc[i, :, b[1]:b[3], b[0]:b[2]]
                    if yc_face.nelement() == 0:
                        print(b)
                        continue
                    yc_face = yc_face.unsqueeze(0)
                    y_face = y[i, :, b[1]:b[3], b[0]:b[2]].unsqueeze(0)

                    yc_face = F.interpolate(
                        yc_face, size=(96, 96), mode="bilinear",
                        align_corners=False)
                    y_face = F.interpolate(
                        y_face, size=(96, 96), mode="bilinear",
                        align_corners=False)

                    facial_loss += self.mse_loss(
                        self.face_recog_model(yc_face),
                        self.face_recog_model(y_face))

        return loss + 1e7 * facial_loss
