import torch
import torchvision.transforms as transforms


class ISM:
    def __init__(self):
        pass

    def __call__(self, image):
        """Reconstruct the ISM image from raw airyscan data

        Parameters
        ----------
        image: ndarray
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        ndarray: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        return torch.sum(image, axis=0)

