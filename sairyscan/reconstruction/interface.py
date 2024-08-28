"""Interface for Airyscan reconstruction process"""
import torch
from sairyscan.core import SObservable


class SAiryscanReconstruction(SObservable):
    """Interface for Airyscan high resolution image reconstruction method"""
    def __call__(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Do the reconstruction

        :param image: Raw detector stack to reconstruct [H (Z) Y X]
        :return: high resolution image [(Z) Y X]
        """
        raise NotImplementedError('SairyscanReconstruction is an interface. Please implement the'
                                  ' __call__ method')

    @staticmethod
    def _crop(image: torch.Tensor) -> torch.Tensor:
        """Crop the image by 5 pixels to avoid side artifacts

        :param image: Image to crop,
        :return: Cropped image
        """
        if image.ndim == 2:
            return image[5:-5, 5:-5]
        return image[5:-5, 5:-5, :]
