"""
Compute the facial loss using facial recognition.

Compute the style and content loss using a VGG-16 model trained on ImageNet.
Uses perceptual loss to compute the style, content loss, and facial loss.
"""
import pathlib
import torch
import torch.nn.functional as F
from PIL import Image

import mtcnn.detector
import openface

import perceptual_loss_net


class FacePerceptualLossNet(PerceptualLossNet):
    """Account for loss in facial recognition."""

    def __init__(self, content_weight, style_weights, regularization_weight,
                 face_weight):
        """Initialize a face detection model on top of the perceptual loss."""
        super(FacePerceptualLossNet, self).__init__(
            content_weight, style_weights, regularization_weight)
        # Save weight for facial loss
        self.face_weight = face_weight
        # Construct models for facial recognition
        self.face_recog_model = openface.net.model
        self.face_recog_model.load_state_dict(
            torch.load(openface.openface_model_path))
        self.face_recog_model.eval()
        self.n_faces_seen = 0

    def compute_perceptual_loss(self, y, yc, ys):
        """Compute the perceptual loss including the face loss."""
        loss = super(FacePerceptualLossNet, self).compute_perceptual_loss(
            y, yc, ys)

        facial_loss = 0.0
        for i, image in enumerate(yc):
            image_array = image.clone().numpy().clip(0, 255)
            pil_image = Image.fromarray(
                image_array.transpose(1, 2, 0).astype("uint8"))
            bounding_boxes, landmarks = mtcnn.detector.detect_faces(pil_image)
            for face_bb in bounding_boxes:
                if face_bb[-1] > 0.9:
                    print("Face found")
                    self.n_faces_seen += 1
                    b = face_bb[:-1].round().astype("int")

                    yc_face = yc[i, :, b[1]:b[3], b[0]:b[2]].unsqueeze(0)
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

        return loss + 1e8 * facial_loss
