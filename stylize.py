"""Stylize an image using the trained image transformation network."""
import pathlib
import argparse
import re

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
    kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}

    # Load the content image to stylize
    print("Loading content image {}".format(args.content_image))
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)])
    # Repeat the image so it matches the batch size for loss computations
    x = content_transform(Image.open(args.content_image))
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        print("Loading the style image transform network, {}".format(
            args.style_model))
        img_transform = ImageTransformNet()
        model_params = torch.load(args.style_model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        img_transform.load_state_dict(model_params)
        img_transform.to(device)
        stylized_img = img_transform(x).cpu()
    stylized_img = stylized_img.squeeze(0).numpy().transpose(1, 2, 0).astype("uint8")
    stylized_img = Image.fromarray(stylized_img)
    stylized_img.save(args.output)


if __name__ == "__main__":
    def main():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Stylize an image using a trained ImageTransformNet")
        parser.add_argument("--content-image", type=pathlib.Path,
                            required=True, metavar="path/to/content/",
                            help="folder where training data is located")
        parser.add_argument("--content-size", type=int, default=256,
                            metavar="N",
                            help=("size to rescale content image(s) to "
                                  "(default: 256 x 256)"))
        parser.add_argument("--model", "--style-model", type=pathlib.Path,
                            required=True, metavar="path/to/model.pth",
                            dest="style_model",
                            help="path to the trained ImageTransformNet")
        parser.add_argument("--output", default=pathlib.Path("out.png"),
                            metavar="path/to/output.ext", type=pathlib.Path,
                            help="what to name the output file")
        parser.add_argument("--no-cuda", action="store_true",
                            help="disables CUDA training")
        args = parser.parse_args()
        print("{}\n".format(args))

        stylize_image(args)
    main()
