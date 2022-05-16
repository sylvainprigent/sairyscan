import torch
from .interface import SAiryscanReconstruction


class ISFED(SAiryscanReconstruction):
    def __init__(self, epsilon=0.3):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, image, reg_image):
        """Reconstruct the ISFED image from raw airyscan data

        Parameters
        ----------
        image: Tensor
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images
        reg_image
            Co-registered airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        out = torch.sum(reg_image, axis=0) - self.epsilon *torch.sum(image, axis=0)
        return torch.nn.functional.relu(out, inplace=True)


metadata = {
    'name': 'ISFED',
    'class': ISFED,
    'parameters': {
        'epsilon': {
            'type': float,
            'label': 'epsilon',
            'help': 'Weighting parameter',
            'default': 0.3
        }
    }
}
