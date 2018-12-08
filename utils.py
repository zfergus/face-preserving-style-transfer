"""Utilties for neural style transfer."""


def gram_matrix(x):
    """Create the gram matrix of x."""
    b, c, h, w = x.shape
    phi = x.view(b, c, h * w)
    return phi.bmm(phi.transpose(1, 2)) / (c * h * w)


def normalize_batch(batch):
    """Normalize using imagenet mean and std."""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch / 255.0
    return (batch - mean) / std
