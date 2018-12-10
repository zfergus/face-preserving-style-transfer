"""Stylize an image using the trained image transformation network."""
import pathlib
import argparse
import re
import sys

import torch
from torchvision import transforms, datasets
import numpy
from PIL import Image

from image_transform_net import ImageTransformNet
import utils


def stylize_image(args):
    """Stylize an image using the trained image transformation network."""
    # Should computation be done on the GPU if available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the content image to stylize
    print("Loading content image ({}) ... ".format(args.content_image), end="")
    content_image = utils.load_image_tensor(
        args.content_image, 1, args.content_shape).to(device)
    print("done")

    with torch.no_grad():
        # Load the style transfer model
        print("Loading the style transfer model ({}) ... ".format(
            args.style_model), end="")
        img_transform = utils.load_model(
            args.style_model, ImageTransformNet()).to(device)
        print("done")

        # Stylize the content image
        print("Stylizing image ... ", end="")
        sys.stdout.flush()
        stylized_img = img_transform(content_image).cpu()
        print("done")

    # Save the stylized image
    print("Saving stylized image to {} ... ".format(args.output), end="")
    utils.save_image_tensor(args.output, stylized_img)
    print("done")


if __name__ == "__main__":
    def main():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Stylize an image using a trained ImageTransformNet")
        parser.add_argument("--content-image", type=pathlib.Path,
                            required=True, metavar="path/to/content/",
                            help="folder where training data is located")
        parser.add_argument("--content-shape", type=int, default=None,
                            metavar="N", nargs=2,
                            help=("size to rescale content image(s) to"))
        parser.add_argument("--model", "--style-model", type=pathlib.Path,
                            required=True, metavar="path/to/model.pth",
                            dest="style_model",
                            help="path to the trained ImageTransformNet")
        parser.add_argument("--output", default=pathlib.Path("out.png"),
                            metavar="path/to/output.ext", type=pathlib.Path,
                            help="what to name the output file")
        parser.add_argument("--no-cuda", action="store_true",
                            help="disables CUDA training")
        stylize_image(parser.parse_args())
    main()
