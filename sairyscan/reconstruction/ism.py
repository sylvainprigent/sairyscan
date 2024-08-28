"""This module implements the ISM with denoising reconstruction method"""
import torch
from .interface import SAiryscanReconstruction


class ISM(SAiryscanReconstruction):
    """Reconstruct a high resolution image by summing the co-registered detectors: ISM method"""
    def __init__(self):
        super().__init__()
        self.num_args = 1

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the reconstruction

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images
        :return: Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        self.progress(0)
        out = torch.sum(image, axis=0)
        self.progress(100)
        return self._crop(out)


metadata = {
    'name': 'ISM',
    'label': 'ISM',
    'class': ISM,
    'parameters': {
        }
}
