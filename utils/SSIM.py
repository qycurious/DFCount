import torch
import torch.nn.functional as F


def ssim(img1, img2, window_size=11, channel=1, size_average=True):
    """
    Calculate SSIM index for a pair of images.

    Args:
        img1 (Tensor): First input image batch. Should be in range [-1, 1].
        img2 (Tensor): Second input image batch. Should be in range [-1, 1].
        window_size (int, optional): Size of the sliding window. Default: 11.
        channel (int, optional): Number of channels in the input images. Default: 1.
        size_average (bool, optional): If True, ssim is calculated as the average of window results.

    Returns:
        Tensor: The calculated SSIM index.
    """
    assert img1.size() == img2.size(), "Input images must have the same dimensions."
    assert img1.dim() == 4, "Input images must be 4D tensors with shape (N, C, H, W)."
    assert channel in [1, 3], "Input images must have 1 or 3 channels."

    padding = window_size // 2
    mu1 = F.pad(img1, (padding, padding, padding, padding), 'reflect')
    mu2 = F.pad(img2, (padding, padding, padding, padding), 'reflect')

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.pad(img1.pow(2), (padding, padding, padding, padding), 'reflect') - mu1_sq
    sigma2_sq = F.pad(img2.pow(2), (padding, padding, padding, padding), 'reflect') - mu2_sq
    sigma12 = F.pad(img1 * img2, (padding, padding, padding, padding), 'reflect') - mu1_mu2

    if channel == 3:
        C1 = 0.01 ** 2 * 3
        C2 = 0.03 ** 2 * 3
    else:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.view(-1)