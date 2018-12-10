"""Train the model image transformation network."""
import pathlib
import argparse

import torch
from torchvision import transforms, datasets

from image_transform_net import ImageTransformNet
from perceptual_loss_net import PerceptualLossNet
import utils


def train(args):
    """Train the model image transformation network."""
    # Should computation be done on the GPU if available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}

    # Load the content images for training
    content_loader = utils.load_content_dataset(
        args.content_images, args.content_size, args.batch_size)

    # Load the style image to train for
    print("Loading style image {}\n".format(args.style_image))
    ys = utils.load_image_tensor(
        args.style_image, args.batch_size, args.style_size).to(device)

    # Newtork to train that stylizes images
    print("Creating image transformation network ... ", end="")
    img_transform = ImageTransformNet().to(device)
    optimizer = torch.optim.Adam(img_transform.parameters(), lr=args.lr)
    print("done")

    print("Creating loss network ... ", end="")
    loss_net = PerceptualLossNet(args.content_weight, args.style_weights,
                                 args.regularization_weight).to(device)
    print("done\n")

    # Load from a checkpoint if necessary
    start_epoch = (utils.load_checkpoint(
        args.checkpoint, img_transform, optimizer, args.lr)
        if args.checkpoint else 1)

    # Begin training the image transform network
    for epoch in range(start_epoch, args.epochs + 1):
        for batch_idx, (yc, _) in enumerate(content_loader):
            optimizer.zero_grad()

            # Stylize the content images
            yc = yc.to(device)
            y = img_transform(yc)

            # Compute the perceptual loss
            loss = loss_net.compute_perceptual_loss(y, yc, ys)

            # Optimize
            loss.backward()
            optimizer.step()

            # Log progress
            if batch_idx % args.log_interval == 0:
                print(("Train Epoch: {:02d} [{:06d}/{:06d} ({:.0f}%)]\t"
                       "Loss: {:12.2f}").format(
                    epoch, batch_idx * len(yc), len(content_loader.dataset),
                    100. * batch_idx / len(content_loader), loss.data.item()))
            # Save checkpoint
            if batch_idx % args.checkpoint_interval == 0:
                utils.save_checkpoint(
                    (args.output_dir / "checkpoint_{:02d}_{:06d}.pth".format(
                            epoch, batch_idx)),
                    epoch, img_transform, optimizer, device)

    # Save a model file to evaluate later
    utils.save_model(
        args.output_dir / "model.pth", img_transform, device)
    utils.save_checkpoint(
        args.output_dir / "final_checkpoint.pth",
        epoch, img_transform, optimizer, device)


if __name__ == "__main__":
    def main():
        """Parse training settings."""
        parser = argparse.ArgumentParser(
            description="Train CNN for Neural Style Transfer")
        parser.add_argument("--content-images", type=pathlib.Path,
                            required=True, metavar="path/to/content/",
                            help="folder where training data is located")
        parser.add_argument("--content-size", type=int, default=256,
                            metavar="N",
                            help=("size to rescale content image(s) to "
                                  "(default: 256 x 256)"))
        parser.add_argument("--content-weight", type=float, default=1e5,
                            help="weight for content loss (default: 1e5)")
        parser.add_argument("--style-image", type=pathlib.Path,
                            required=True, metavar="path/to/style.image",
                            help="path to the style image to train for")
        parser.add_argument("--style-size", type=int, default=None,
                            metavar="N", nargs=2,
                            help=("size to rescale the style image to "
                                  "(default: unscaled)"))
        parser.add_argument("--style-weights", type=float,
                            default=[1e10, 1e10, 1e10, 1e10, 1e10], nargs=5,
                            help="weight for style loss (default: 1e10)")
        parser.add_argument("--regularization-weight", type=float,
                            default=1e-3,
                            help="weight for regularized TV (default: 1e-3)")
        parser.add_argument("--output-dir", default=pathlib.Path("."),
                            metavar="path/to/output/", type=pathlib.Path,
                            help="where to store model and checkpoint files")
        parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                            help="input batch size for training (default: 4)")
        parser.add_argument("--epochs", type=int, default=2, metavar="N",
                            help="number of epochs to train (default: 2)")
        parser.add_argument("--lr", type=float, default=1e-3, metavar="LR",
                            help="learning rate (default: 1e-3)")
        parser.add_argument("--log-interval", type=int, default=1,
                            metavar="N", help="how many batches between logs")
        parser.add_argument("--checkpoint", default=None, type=pathlib.Path,
                            metavar="path/to/checkpoint.pth",
                            help="checkpoint file to continue training")
        parser.add_argument("--checkpoint-interval", default=5000, type=int,
                            metavar="N",
                            help="how many batches between checkpoint saves")
        parser.add_argument("--no-cuda", action="store_true",
                            help="disables CUDA training")
        args = parser.parse_args()
        print("{}\n".format(args))

        args.output_dir.mkdir(parents=True,  exist_ok=True)
        train(args)
    main()
