"""Interface for a Airyscan registration filter

Classes
-------
SairyscanRegistration

"""


class SairyscanRegistration:
    """Interface for Airyscan detectors registration method"""
    def __init__(self):
        pass

    def __call__(self, image):
        """Do the registration

        Parameters
        ----------
        image: Tensor
            Raw airyscan data for a single channel time point [H (Z) Y X]

        Return
        ------
        Tensor: co-registered detectors [H (Z) Y X]

        """
        raise NotImplementedError('SairyscanRegistration is an interface. Please implement the'
                                  ' __call__ method')
