"""Interface for a Airyscan enhancing filter

Classes
-------
SAiryscanRegistration

"""
from sairyscan.core import SObservable


class SAiryscanEnhancing(SObservable):
    """Interface for Airyscan enhancing filter applied after the reconstruction"""
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        """Do the enhancing

        Parameters
        ----------
        image: Tensor
            Reconstructed airyscan data for a single channel time point [(Z) Y X]

        Return
        ------
        Tensor: co-registered detectors [(Z) Y X]

        """
        raise NotImplementedError('SairyscanEnhancing is an interface. Please implement the'
                                  ' __call__ method')
