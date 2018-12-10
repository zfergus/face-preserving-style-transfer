"""Utilties for neural style transfer."""
import pathlib
import numpy
import torch
from torchvision import transforms, datasets
from PIL import Image
import cv2 as cv


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


def save_checkpoint(filename, epoch, model, optimizer, device="cpu"):
    """
    Save a checkpoint to, potentially, continue training.

    Different from the model file because the optimizer's state is also saved.
    """
    model.eval().cpu()
    checkpoint = {"epoch": epoch, "model": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(filename))
    print(("Saved checkpoint to {0}. You can run "
           "`python train.py --checkpoint {0}` to continue training from "
           "this state.").format(filename))
    model.to(device).train()


def load_model(filename, model):
    """Load the model parameters in {filename}."""
    model_params = torch.load(filename)
    model.load_state_dict(model_params)
    return model


def save_model(filename, model, device="cpu"):
    """Save the model weights."""
    model.eval().cpu()
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(filename))
    print(("Saved model to {0}. You can run "
           "`python stylize.py --model {0}` to stylize an image").format(
           model_file))
    model.to(device).train()


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


def load_image_tensor(filename, batch_size, image_shape=None):
    """Load an image for torch."""
    image = Image.open(filename)
    if image_shape:  # Downsample the image
        image = image.resize(image_shape, Image.ANTIALIAS)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)])
    # Repeat the image so it matches the batch size for loss computations
    return image_transform(image).repeat(batch_size, 1, 1, 1)


def save_image_tensor(filename, image_tensor):
    """Save a tensor of an image."""
    image = Image.fromarray(
        image_tensor.squeeze(0).numpy().transpose(1, 2, 0).astype("uint8"))
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    image.save(filename)


def video_loader(filename, batch_size):
    """Load a video for torch."""
    cap = cv.VideoCapture(str(filename))
    frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    buf = numpy.empty((batch_size, frameHeight, frameWidth, 3), dtype='uint8')

    fc = 0
    ret = True

    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1
        yield torch.tensor(buf.transpose(0, 3, 2, 1)).float()

    cap.release()


def save_video_tensor(filename, video_tensor):
    # initialize video writer
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 30
    width, height = video_tensor.shape[2:]
    out = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

    for frame in video_tensor:
        out.write(frame)

    # close out the video writer
    out.release()
