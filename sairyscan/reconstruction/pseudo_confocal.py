"""This module implements the pseudo-confocal with denoising reconstruction method"""
import torch
from .interface import SAiryscanReconstruction


class PseudoConfocal(SAiryscanReconstruction):
    """Reconstruct an image equivalent to a confocal image by summing the inner detectors

    :param pinhole: Size of the pinhole in AU: [0.6, 1, 1.25]
    """
    def __init__(self, pinhole=1):
        super().__init__()
        self.num_args = 1
        self.pinhole = pinhole
        if pinhole not in [0.6, 1, 1.25]:
            raise ValueError('PseudoConfocal pinhole parameter must be a value in (0.6, 1, 1.25)')

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the reconstruction

        :param image: Raw detector stack to reconstruct [H (Z) Y X]
        :return: high resolution image [(Z) Y X]
        """
        self.progress(0)
        out = None
        if self.pinhole == 0.6:
            out = torch.sum(image[0:7, ...], axis=0)
        elif self.pinhole == 1:
            out = torch.sum(image[0:19, ...], axis=0)
        elif self.pinhole == 1.25:
            out = torch.sum(image[0:32, ...], axis=0)
        self.progress(100)
        return self._crop(out)


metadata = {
    'name': 'PseudoConfocal',
    'label': 'Pseudo Confocal',
    'class': PseudoConfocal,
    'parameters': {
        'pinhole': {
            'type': 'select',
            'label': 'Pinhole',
            'values': [0.6, 1, 1.25],
            'help': 'Size of the pinhole (0.6, 1, 1.25)',
            'default': 0.6
        }
    }
}
