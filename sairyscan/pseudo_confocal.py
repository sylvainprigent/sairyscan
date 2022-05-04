import torch


class PseudoConfocal:
    def __init__(self, pinhole=1):
        self.pinhole = pinhole
        if pinhole not in [0.6, 1, 1.25]:
            raise ValueError('PseudoConfocal pinhole parameter must be a value in (0.6, 1, 1.25)')

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
        if self.pinhole == 0.6:
            return torch.sum(image[0:7, ...], axis=0)
        elif self.pinhole == 1:
            return torch.sum(image[0:19, ...], axis=0)
        elif self.pinhole == 1.25:
            return torch.sum(image[0:32, ...], axis=0)
