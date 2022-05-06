import torch
from .interface import SAiryscanReconstruction


class IFED(SAiryscanReconstruction):
    def __init__(self, inner_ring_index=7, epsilon=0.3):
        super().__init__()
        self.inner_ring_index = inner_ring_index
        self.epsilon = epsilon
        if inner_ring_index not in [7, 19]:
            raise ValueError('Inner ring index must be in (7, 19)')

    def __call__(self, image):
        """Reconstruct the IFED image from raw airyscan data

        Parameters
        ----------
        image: Tensor
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        out = torch.sum(image[0:self.inner_ring_index, ...], axis=0) - self.epsilon * torch.sum(
            image[self.inner_ring_index + 1:32, ...], axis=0)
        return torch.nn.functional.relu(out, inplace=True)