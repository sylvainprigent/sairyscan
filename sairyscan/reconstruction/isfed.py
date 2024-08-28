"""This module implements the ISFED reconstruction method"""
import torch
from .interface import SAiryscanReconstruction
from ._sure import SureMap


class ISFED(SAiryscanReconstruction):
    """Implementation of the ISFED reconstruction method

    :param epsilon: Weight between the two images to combine
    """
    def __init__(self, epsilon=0.3):
        super().__init__()
        self.num_args = 2
        self.epsilon = epsilon

    def __call__(self, image: torch.Tensor, reg_image: torch.Tensor) -> torch.Tensor:
        """Do the reconstruction

        :param image: Raw detector stack to reconstruct [H (Z) Y X]
        :param reg_image: Spatially co-registered detectors stack [H (Z) Y X]
        :return: high resolution image [(Z) Y X]
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
        return self._crop(torch.nn.functional.relu(out, inplace=True))


metadata = {
    'name': 'ISFED',
    'label': 'ISFED',
    'class': ISFED,
    'parameters': {
        'epsilon': {
            'type': str,
            'label': 'epsilon',
            'help': 'Weighting parameter',
            'default': 0.3
        }
    }
}
