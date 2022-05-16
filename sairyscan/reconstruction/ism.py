import torch
from .interface import SAiryscanReconstruction


class ISM(SAiryscanReconstruction):
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        """Reconstruct the ISM image from raw airyscan data

        Parameters
        ----------
        image: Tensor
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        return torch.sum(image, axis=0)


metadata = {
    'name': 'ISM',
    'class': ISM,
    'parameters': {
        }
}