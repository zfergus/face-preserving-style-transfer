"""
Compute the face loss using facial recognition.

Compute the style and content loss using a VGG-16 model trained on ImageNet.
Uses perceptual loss to compute the style, content loss, and face loss.
"""
import numpy
import torch
import torch.nn.functional as F
from PIL import Image

from . import mtcnn
from . import openface

from . import perceptual_loss_net


class FacePerceptualLossNet(perceptual_loss_net.PerceptualLossNet):
    """Account for loss in facial recognition."""

    def __init__(self, content_weight, style_weights, regularization_weight,
                 face_weight):
        """Initialize a face detection model on top of the perceptual loss."""
        super(FacePerceptualLossNet, self).__init__(
            content_weight, style_weights, regularization_weight)
        # Save weight for face loss
        self.face_weight = face_weight
        # Construct models for facial recognition
        self.face_recog_model = openface.net.model
        self.face_recog_model.load_state_dict(
            torch.load(openface.openface_model_path))
        self.face_recog_model.eval()
        self.n_faces_seen = 0

    def compute_face_perceptual_loss(self, y, yc):
        """Compute the face loss by using mtcnn and openface."""
        face_loss = 0.0
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
                        continue
                    yc_face = yc_face.unsqueeze(0)
                    y_face = y[i, :, b[1]:b[3], b[0]:b[2]].unsqueeze(0)

                    yc_face = F.interpolate(
                        yc_face, size=(96, 96), mode="bilinear",
                        align_corners=False)
                    y_face = F.interpolate(
                        y_face, size=(96, 96), mode="bilinear",
                        align_corners=False)

                    face_loss += self.mse_loss(
                        self.face_recog_model(yc_face),
                        self.face_recog_model(y_face))
        return self.face_weight + face_loss

    def compute_perceptual_loss(self, y, yc, ys):
        """Compute the perceptual loss including the face loss."""
        return super(FacePerceptualLossNet, self).compute_perceptual_loss(
            y, yc, ys) + self.compute_face_perceptual_loss(y, yc)
