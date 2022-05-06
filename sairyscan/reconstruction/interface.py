"""Interface for a Airyscan reconstruction filter

Classes
-------
SairyscanReconstruction

"""


class SAiryscanReconstruction:
    """Interface for Airyscan high resolution image reconstruction method"""
    def __init__(self):
        pass

    def __call__(self, *args):
        """Do the reconstruction

        Parameters
        ----------
        args: list of Tensor
            Raw airyscan data for a single channel time point [H (Z) Y X]. It can be one image (raw
            image) or several images (rax image, co-registered detectors image)

        Return
        ------
        Tensor: high resolution image[(Z) Y X]

        """
        raise NotImplementedError('SairyscanReconstruction is an interface. Please implement the'
                                  ' __call__ method')
