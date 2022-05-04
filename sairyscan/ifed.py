import torch
import torchvision.transforms as transforms


class IFED:
    def __init__(self, inner_ring_index=7, epsilon=0.3, smooth=False):
        self.inner_ring_index = inner_ring_index
        self.epsilon = epsilon
        self.smooth = smooth
        if inner_ring_index not in [7, 19]:
            raise ValueError('Inner ring index must be in (7, 19)')

    def __call__(self, image):
        """Reconstruct the pseudo confocal image from raw airyscan data

        Parameters
        ----------
        image: ndarray
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        ndarray: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        out = torch.sum(image[0:self.inner_ring_index, ...], axis=0) - self.epsilon * torch.sum(
            image[self.inner_ring_index + 1:32, ...], axis=0)
        out = torch.nn.functional.relu(out, inplace=True)
        if self.smooth:
            out_bc = out.view(1, 1, out.shape[0], out.shape[1])
            out_b = transforms.functional.gaussian_blur(out_bc, kernel_size=(11, 11), sigma=0.5)
            return out_b.view(out.shape)
        else:
            return out
