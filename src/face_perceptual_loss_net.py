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

    def compute_face_perceptual_loss(self, y, yc):
        """Compute the face loss by using mtcnn and openface."""
        face_loss = 0.0  # 0 if no faces found
        for i, image in enumerate(yc):  # Find faces per image in the batch
            # Convert the content image from a tensor to a PIL Image
            image_array = image.cpu().numpy().clip(0, 255)
            pil_image = Image.fromarray(
                image_array.transpose(1, 2, 0).astype("uint8"))
            # MTCCN takes a PIL vector so this breaks the autograd, hence face
            # detection is not differentiable at the moment. This may cause
            # suboptimial results because the bounding boxes for face
            # recognition are not differentiable.
            # TODO: Improve the MTCNN implementation by taking a torch.Tensor
            #       to allow MTCNN to be autmoatically differentiated.
            bounding_boxes, landmarks = mtcnn.detector.detect_faces(pil_image)
            # Loop over all the faces found bby MTCNN
            for face_bb in bounding_boxes:
                # Only consider faces with high probability
                if face_bb[-1] > 0.9:
                    # Round the bounding box to integer indices
                    b = face_bb[:-1].round().astype("int")
                    b[::2] = numpy.clip(b[::2], 0, yc.shape[2])
                    b[1::2] = numpy.clip(b[1::2], 0, yc.shape[3])

                    # Get the face patch from the content tensor
                    yc_face = yc[i, :, b[1]:b[3], b[0]:b[2]]
                    if yc_face.nelement() == 0:  # No pixels selected
                        continue
                    yc_face = yc_face.unsqueeze(0)  # Remove batch dim
                    # Get the face patch from the stylized tensor
                    y_face = y[i, :, b[1]:b[3], b[0]:b[2]].unsqueeze(0)

                    # Downsample the face to a 96 x 96 patch
                    yc_face = F.interpolate(
                        yc_face, size=(96, 96), mode="bilinear",
                        align_corners=False)
                    y_face = F.interpolate(
                        y_face, size=(96, 96), mode="bilinear",
                        align_corners=False)
                    # Face loss is the mean squared error of the OpenFace
                    # face descriptors
                    face_loss += self.mse_loss(
                        self.face_recog_model(yc_face),
                        self.face_recog_model(y_face))
        # Weight the face loss
        return self.face_weight * face_loss

    def compute_perceptual_loss(self, y, yc, ys):
        """Compute the perceptual loss including the face loss."""
        return super(FacePerceptualLossNet, self).compute_perceptual_loss(
            y, yc, ys) + self.compute_face_perceptual_loss(y, yc)
