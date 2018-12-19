"""Stylize an image using the trained image transformation network."""
import pathlib
import argparse
import sys
import torch

from .image_transform_net import ImageTransformNet
from . import utils


def stylize(args):
    """Stylize an image(s) using the trained image transformation network."""
    # Should computation be done on the GPU if available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the content image to stylize
    if args.video:
        args.output = args.output.with_suffix(".mkv")
        video = utils.VideoReaderWriter(args.content_file, args.output, 4)
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
        if args.video:
            for i, frames in enumerate(video.frames()):
                stylized_frames = img_transform(frames.to(device)).cpu()
                video.write(stylized_frames)
                if i % 1 == 0:
                    current_frame = max((i + 1) * 4, video.frame_count)
                    print(("Saved frame: {:04d}/{:04d} ({:.0f}%)").format(
                        current_frame, video.frame_count,
                        100. * current_frame / video.frame_count))
        else:
            print("Stylizing image ... ", end="")
            sys.stdout.flush()
            stylized_img = img_transform(content_image).cpu()
            print("done")
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
        args = parser.parse_args()
        args.video &= "cv2" in sys.modules
        stylize(args)
    main()
