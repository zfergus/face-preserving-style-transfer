"""Train the model image transformation network."""
import pathlib
import argparse

import torch
from torchvision import transforms, datasets
import numpy
from PIL import Image

from image_transform_net import ImageTransformNet
from loss_net import LossNet
import utils


def train(args):
    """Train the model image transformation network."""
    # Should computation be done on the GPU if available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}

    # Load the content images for training
    # Image values will be floating points in the range [0, 1]
    print("Creating dataset for content images in {}".format(
        args.content_images))
    content_transform = transforms.Compose([
        transforms.Resize(args.content_size),
        transforms.CenterCrop(args.content_size),
        transforms.ToTensor()])
    content_data = datasets.ImageFolder(
        str(args.content_images), content_transform)
    content_loader = torch.utils.data.DataLoader(
        content_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Load the style image to train for
    print("Loading style image {}".format(args.style_image))
    style_image = Image.open(args.style_image)
    if args.style_size:
        # Downsample the image
        style_image.resize(args.style_size, Image.ANTIALIAS)
    style_transform = transforms.ToTensor()
    # Repeat the image so it matches the batch size for loss computations
    ys = style_transform(Image.open(args.style_image)).repeat(
        args.batch_size, 1, 1, 1).to(device)

    # Newtork to train that stylizes images
    print("Creating image transformation network")
    img_transform = ImageTransformNet().to(device)
    # Pretrained VGG network to return the relu values
    print("Creating loss network")
    loss_net = LossNet().to(device)
    optimizer = torch.optim.Adam(img_transform.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    # Precompute the Loss Network features of the style image
    print("Computing features and gram matrices of style image")
    ys_features = loss_net(utils.normalize_batch(ys))
    # Precompute the gram matrices of the style features, used for style loss
    ys_grams = [utils.gram_matrix(feature) for feature in ys_features]

    # Load from a checkpoint if necessary
    start_epoch = 1
    if args.checkpoint is not None:
        print("Loading checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        img_transform.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        print("Continuing training from checkpoint "
              "{:s} at epoch {:d}\n".format(args.checkpoint, start_epoch))

    # Begin training the image transform network
    for epoch in range(start_epoch, args.epochs + 1):
        img_transform.to(device).train()
        for batch_idx, (yc, _) in enumerate(content_loader):
            optimizer.zero_grad()
            yc = yc.to(device)

            # Stylize the content images
            y = img_transform(yc)

            # Compute the Loss Network features of the contnent and stylized
            # content.
            yc_features = loss_net(utils.normalize_batch(yc))
            y_features = loss_net(utils.normalize_batch(y))

            # Feature loss is the mean squared error of the content and
            # stylized content
            feature_loss = mse_loss(yc_features.relu3_3, y_features.relu3_3)

            # Style loss id the Frobenius norm of the gram matrices
            style_loss = sum([style_weight * mse_loss(
                utils.gram_matrix(y_feature), ys_gram[:yc.shape[0]])
                for y_feature, ys_gram, style_weight in zip(
                    y_features, ys_grams, args.style_weights)])

            # Compute the regularized total variation of the stylized image
            total_variation = (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            # The total loss is a weighted sum of the loss values
            loss = (args.feature_weight * feature_loss + style_loss +
                    args.regularization_weight * total_variation)
            # Optimize
            loss.backward()
            optimizer.step()

            # Log progress
            if batch_idx % args.log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(yc), len(content_loader.dataset),
                    100. * batch_idx / len(content_loader), loss.data.item()))
        # Save a model file to evaluate later
        print("Saving checkpoint and model file")
        img_transform.eval().cpu()
        model_file = args.output_dir / "model_{:02d}.pth".format(epoch)
        torch.save(img_transform.state_dict(), str(model_file))
        print(("Saved model to {0}. You can run "
                "`python stylize.py --model {0}` to continue stylize an image "
                "this state.\n").format(model_file))

        # Save a checkpoint to, potentially, continue training. Different from
        # the model file because the optimizer's state is also saved
        checkpoint_file = args.output_dir / "checkpoint_{:02d}.pth".format(epoch)
        checkpoint = {"epoch": epoch, "model": img_transform.state_dict(),
                      "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, str(checkpoint_file))
        print(("Saved checkpoint to {0}. You can run "
               "`python train.py --checkpoint {0}` to continue training from "
               "this state.\n").format(checkpoint_file))


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
        parser.add_argument("--feature-weight", type=float, default=1e5,
                            help="weight for feature loss (default: 1e0)")
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
                            default=1e0,
                            help="weight for regularized TV (default: 1e-6)")
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
        parser.add_argument("--no-cuda", action="store_true",
                            help="disables CUDA training")
        args = parser.parse_args()
        print("{}\n".format(args))

        args.output_dir.mkdir(parents=True, exist_ok=True)
        train(args)
    main()
