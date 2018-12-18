"""Test the face stylization performance."""
import pathlib
import argparse
import sys
import torch
import numpy

from .image_transform_net import ImageTransformNet
from .face_perceptual_loss_net import FacePerceptualLossNet
from . import utils


def test(args):
    """Stylize an image(s) using the trained image transformation network."""
    # Should computation be done on the GPU if available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the content images for testing
    content_loader = utils.load_content_dataset(
        args.content_images, args.content_size, 1)
    print("Loaded dataset images ({})".format(args.content_images))

    with torch.no_grad():
        # Load the style transfer model
        print("Loading the base style transfer model ({}) ... ".format(
            args.base_model), end="")
        base_img_transform = utils.load_model(
            args.base_model, ImageTransformNet()).to(device)
        print("done")

        print(("Loading the face preserving style transfer model ({}) "
               "... ").format(args.face_model), end="")
        face_img_transform = utils.load_model(
            args.face_model, ImageTransformNet()).to(device)
        print("done")

        print("Creating loss network ... ", end="")
        loss_net = FacePerceptualLossNet(0, 0, 0, 1).to(device)
        print("done\n")

        # Begin testing the image transform network
        print("Testing model:")
        sys.stdout.flush()
        base_losses = numpy.full(len(content_loader.dataset), numpy.inf)
        test_losses = numpy.full(len(content_loader.dataset), numpy.inf)
        for i, (yc, _) in enumerate(content_loader):
            yc = yc.to(device)
            y = base_img_transform(yc)
            base_losses[i] = loss.compute_face_perceptual_loss(
                y, yc).data.item()
            y = face_img_transform(yc)
            face_losses[i] = loss.compute_face_perceptual_loss(
                y, yc).data.item()
            print("{:06d} / {:06d} ({:.0f})".format(
                i, len(content_loader.dataset),
                100. * i / len(content_loader.dataset)), end="\r")
        print("done")
    numpy.savez(args.output, base_losses=base_losses, face_losses=face_losses)


if __name__ == "__main__":
    def main():
        """Parse training settings."""
        parser = argparse.ArgumentParser(
            description="Test a model on facial presevation.")
        parser.add_argument("--face-images", type=pathlib.Path,
                            required=True, metavar="path/to/faces/",
                            help="folder where testing data is located")
        parser.add_argument("--face-size", type=int, default=256,
                            metavar="N",
                            help=("size to rescale face image(s) to "
                                  "(default: 256 x 256)"))
        parser.add_argument("--base-model", type=pathlib.Path,
                            required=True, metavar="path/to/base-model.pth",
                            help="path to the base trained ImageTransformNet")
        parser.add_argument("--face-model", type=pathlib.Path,
                            required=True, metavar="path/to/face-model.pth",
                            help="path to the face trained ImageTransformNet")
        parser.add_argument("--output",
                            default=pathlib.Path("test-losses.npz"),
                            metavar="path/to/output.npz", type=pathlib.Path,
                            help="where to store results as an npz file")
        parser.add_argument("--no-cuda", action="store_true",
                            help="disables CUDA training")
        args = parser.parse_args()
        print("{}\n".format(args))

        args.output.resolve().parent.mkdir(parents=True,  exist_ok=True)
        test(args)
    main()
