"""Interface for Airyscan enhancing filter"""
import torch

from sairyscan.core import SObservable


class SAiryscanEnhancing(SObservable):
    """Interface for Airyscan enhancing filter applied after the reconstruction"""
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the enhancing

        :param image: Reconstructed airyscan data for a single channel time point [(Z) Y X]
        :return: Filtered image
        """
        raise NotImplementedError('SairyscanEnhancing is an interface. Please implement the'
                                  ' __call__ method')
