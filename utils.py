"""Utilties for neural style transfer."""

import torch
from torchvision import transforms, datasets
from PIL import Image


def save_checkpoint(filename, epoch, model, optimizer, device="cpu"):
    """
    Save a checkpoint to, potentially, continue training.

    Different from the model file because the optimizer's state is also saved.
    """
    model.eval().cpu()
    checkpoint = {"epoch": epoch, "model": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, str(filename))
    print(("Saved checkpoint to {0}. You can run "
           "`python train.py --checkpoint {0}` to continue training from "
           "this state.").format(filename))
    model.to(device).train()


def load_checkpoint(filename, model, optimizer, lr):
    """Load from a checkpoint."""
    print("Loading checkpoint {}".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("Continuing training from checkpoint "
              "{:s} at epoch {:d}\n".format(filename, start_epoch))
    return start_epoch


def save_model(filename, model, device="cpu"):
    """Save the model weights."""
    model.eval().cpu()
    torch.save(model.state_dict(), str(filename))
    print(("Saved model to {0}. You can run "
           "`python stylize.py --model {0}` to stylize an image").format(
           model_file))
    model.to(device).train()


def load_model(filename, model):
    """Load the model parameters in {filename}."""
    model_params = torch.load(filename)
    model.load_state_dict(model_params)
    return model


def load_content_dataset(content_path, content_size, batch_size):
    """
    Load the content images for training.

    Image values will be floating points in the range [0, 255]
    """
    print("Creating dataset for content images in {}".format(content_path))
    content_transform = transforms.Compose([
        transforms.Resize(content_size),
        transforms.CenterCrop(content_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)])
    content_data = datasets.ImageFolder(str(content_path), content_transform)
    return torch.utils.data.DataLoader(content_data, batch_size=batch_size)


def load_image_tensor(filename, batch_size, image_size=None):
    """Load an image for torch."""
    image = Image.open(filename)
    if image_size:  # Downsample the image
        image.resize(image_size, Image.ANTIALIAS)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)])
    # Repeat the image so it matches the batch size for loss computations
    return image_transform(image).repeat(batch_size, 1, 1, 1)


def save_image_tensor(filename, image_tensor):
    """Save a tensor of an image."""
    image = Image.fromarray(
        image_tensor.squeeze(0).numpy().transpose(1, 2, 0).astype("uint8"))
    image.save(filename)
