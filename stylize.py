"""Stylize an image using the trained image transformation network."""
import pathlib
import argparse
import re
import sys

import torch
from torchvision import transforms, datasets
import numpy
from PIL import Image
import cv2 as cv

from image_transform_net import ImageTransformNet
import utils


def stylize(args):
    """Stylize an image(s) using the trained image transformation network."""
    # Should computation be done on the GPU if available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the content image to stylize
    if args.video:
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fps = 30
        width, height = 1280, 720
        out = cv.VideoWriter(str(args.output), fourcc, fps, (width, height))
    else:
        content_image = utils.load_image_tensor(
            args.content_file, 1, args.content_shape).to(device)
    print("Loaded content {} ({})".format(
        "video" if args.video else "image", args.content_file))

    with torch.no_grad():
        # Load the style transfer model
        print("Loading the style transfer model ({}) ... ".format(
            args.style_model), end="")
        img_transform = utils.load_model(
            args.style_model, ImageTransformNet()).to(device)
        print("done")

        # Stylize the content image
        if(args.video):
            for frames in utils.video_loader(args.content_file, 4):
                stylized_img = img_transform(frames).cpu()
                for frame in stylized_img:
                    out.write(frame.numpy())
            out.release()  # close out the video writer
        else:
            print("Stylizing image ... ", end="")
            sys.stdout.flush()
            stylized_img = img_transform(content_image).cpu()
            print("done")

    # Save the stylized image
    if not args.video:
        utils.save_image_tensor(args.output, stylized_img)
    print("Saved stylized {} to {}".format(
        "video" if args.video else "image", args.output))


if __name__ == "__main__":
    def main():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Stylize an image using a trained ImageTransformNet")
        parser.add_argument("--content-image", type=pathlib.Path,
                            dest="content_file",
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
        parser.add_argument("--video", action="store_true",
                            help="stylize a video")
        stylize(parser.parse_args())
    main()
