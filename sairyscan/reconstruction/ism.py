import torch
from .interface import SAiryscanReconstruction


class ISM(SAiryscanReconstruction):
    def __init__(self):
        super().__init__()
        self.num_args = 1

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
        self.progress(0)
        out = torch.sum(image, axis=0)
        self.progress(100)
        return out


metadata = {
    'name': 'ISM',
    'label': 'ISM',
    'class': ISM,
    'parameters': {
        }
}
