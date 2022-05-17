import torch
from .interface import SAiryscanReconstruction
from ._sure import SureMap


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
        a = torch.sum(reg_image, axis=0)
        b = torch.sum(image, axis=0)

        if self.epsilon == 'map':
            map_ = SureMap(smooth=True)
            epsilon = map_(image[0, ...], a, b)
            print('epsilon map=', epsilon.shape)
        elif self.epsilon == 'mode':
            map_ = SureMap(smooth=False)
            epsilon = map_.mode(image[0, ...], a, b)
            print('mode epsilon=', epsilon)
        else:
            epsilon = self.epsilon
        out = a - epsilon * b
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
