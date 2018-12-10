"""Utilties for neural style transfer."""
import pathlib
import numpy
import torch
from torchvision import transforms, datasets
from PIL import Image
import cv2 as cv

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]


def load_checkpoint(filename, model, optimizer, lr):
    """Load from a checkpoint."""
    print("Loading checkpoint {}".format(filename))
    checkpoint = torch.load(filename, map_location="cpu")
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
    model_params = torch.load(str(filename))
    model.load_state_dict(model_params)
    return model


def save_model(filename, model, device="cpu"):
    """Save the model weights."""
    model.eval().cpu()
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(filename))
    print(("Saved model to {0}. You can run "
           "`python stylize.py --model {0}` to stylize an image").format(
           filename))
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
        transforms.Normalize(mean=image_net_mean, std=image_net_std)])
    content_data = datasets.ImageFolder(str(content_path), content_transform)
    return torch.utils.data.DataLoader(content_data, batch_size=batch_size)


def load_image_tensor(filename, batch_size, image_shape=None):
    """Load an image for torch."""
    image = Image.open(filename)
    if image_shape:  # Downsample the image
        image = image.resize(image_shape, Image.ANTIALIAS)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=image_net_mean, std=image_net_std)])
    # Repeat the image so it matches the batch size for loss computations
    return image_transform(image).repeat(batch_size, 1, 1, 1)


def save_image_tensor(filename, image_tensor):
    """Save a tensor of an image."""
    image_array = image_tensor.clone().squeeze(0).numpy()
    image_array *= np.array(image_net_std).reshape(3, 1, 1)
    image_array += np.array(image_net_mean).reshape(3, 1, 1)
    image_array *= 255
    image = Image.fromarray(
        image_array.clip(0, 255).transpose(1, 2, 0).astype("uint8"))
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    image.save(filename)


class VideoReaderWriter:
    """Class to read and write a video file with the same properties."""

    def __init__(self, in_file, out_file, batch_size=16):
        """Open the input video and output video."""
        self.in_video = cv.VideoCapture(str(in_file))
        self.frame_count = int(self.in_video.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.in_video.get(cv.CAP_PROP_FPS))
        self.frame_width = int(self.in_video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.in_video.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.batch_size = batch_size
        self.buf = numpy.empty(
            (self.batch_size, self.frame_height, self.frame_width, 3),
            dtype='uint8')

        fourcc = cv.VideoWriter_fourcc(*'MPEG')
        pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        self.out_video = cv.VideoWriter(str(out_file), fourcc,
                                        self.fps,
                                        (self.frame_width, self.frame_height))

    def frames(self):
        """
        Get the frames of the input as a generator.

        Closes both videos upon completion.
        """
        fc = 0
        ret = True

        while fc < self.frame_count and ret:
            nFrames = min(self.batch_size, self.frame_count - fc)
            for i in range(nFrames):
                ret, buffer_i = self.in_video.read()
                if buffer_i is not None:
                    self.buf[i] = buffer_i
                else:
                    nFrames = i
                    ret = False
                    break
                fc += 1
            yield torch.tensor(
                self.buf[:nFrames].transpose(0, 3, 1, 2)).float()

        self.close()

    def write(self, frames):
        """Write frames to the output video."""
        for frame in frames:
            self.out_video.write(
                frame.numpy().transpose(1, 2, 0).astype("uint8"))

    def close(self):
        """Close the input and output videos."""
        self.in_video.release()
        self.out_video.release()
