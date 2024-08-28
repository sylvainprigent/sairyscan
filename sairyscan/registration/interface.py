"""Interface for Airyscan registration filter"""
import torch
from ..core import SObservable


class SAiryscanRegistration(SObservable):
    """Interface for Airyscan detectors registration method"""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the registration

        :param image: Raw airyscan data for a single channel time point [H (Z) Y X]
        :return: co-registered detectors [H (Z) Y X]
        """
        raise NotImplementedError('SairyscanRegistration is an interface. Please implement the'
                                  ' __call__ method')
