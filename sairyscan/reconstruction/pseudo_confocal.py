import torch
from .interface import SAiryscanReconstruction


class PseudoConfocal(SAiryscanReconstruction):
    def __init__(self, pinhole=1):
        super().__init__()
        self.pinhole = pinhole
        if pinhole not in [0.6, 1, 1.25]:
            raise ValueError('PseudoConfocal pinhole parameter must be a value in (0.6, 1, 1.25)')

    def __call__(self, image):
        """Reconstruct the pseudo confocal image from raw airyscan data

        Parameters
        ----------
        image: Tensor
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        if self.pinhole == 0.6:
            return torch.sum(image[0:7, ...], axis=0)
        elif self.pinhole == 1:
            return torch.sum(image[0:19, ...], axis=0)
        elif self.pinhole == 1.25:
            return torch.sum(image[0:32, ...], axis=0)


metadata = {
    'name': 'PseudoConfocal',
    'class': PseudoConfocal,
    'parameters': {
        'pinhole': {
            'type': float,
            'label': 'Pinhole',
            'help': 'Size of the pinhole (0.6, 1, 1.25)',
            'default': 0.6
        }
    }
}
