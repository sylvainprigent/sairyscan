"""This module implements the ISFED with denoising reconstruction method"""
import torch

from ..enhancing.spitfire_denoise import SpitfireDenoise

from .interface import SAiryscanReconstruction
from ._sure import SureMap
from .spitfire_join_denoise import SpitfireJoinDenoise


class ISFEDDenoising(SAiryscanReconstruction):
    """Reconstruct a high resolution image with a ISFED method including a denoising step

    The denoising is performed on the two terms of the ISFED image difference using the SPITFIR(e)
    algorithm

    :param epsilon: weighting parameter for the ISFED difference second term. If epsilon='map',
        epsilon is an automatic estimated weight map using the SURE criterion. If epsilon='mode',
        epsilon is a float which corresponds to the main mode of the SURE map. Otherwise, epsilon
        can be fixed to any float value
    :param reg_inner: Regularization for denoising of the first term of ISFED reconstruction
    :param reg_outer: Regularization for the denoising of the second term of the SFED reconstruction
    :param weighting_inner: Weighting parameter of the SPITFIR(e) denoising model for the first
        term of ISFED. Must be in [0, 1], with value close to 0 for sparse signal and close to
        one otherwise.
    :param weighting_outer: Weighting parameter of the SPITFIR(e) denoising model for the second
        term of ISFED. Must be in [0, 1], with value close to 0 for sparse signal and close to one
        otherwise.
    """
    def __init__(self,
                 epsilon: float = 0.3,
                 reg_inner: float = 0.995,
                 reg_outer: float = 0.995,
                 weighting_inner: float = 0.9,
                 weighting_outer: float = 0.9,
                 join_denoising: bool = False):
        super().__init__()
        self.num_args = 2
        self.epsilon = epsilon
        self.reg_inner = reg_inner
        self.reg_outer = reg_outer
        self.weighting_inner = weighting_inner
        self.weighting_outer = weighting_outer
        self.join_denoising = join_denoising
        self.map_ = None

    def __call__(self, image: torch.Tensor, reg_image: torch.Tensor) -> torch.Tensor:
        """Do the reconstruction

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images,
        :param reg_image: Co-registered airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for
            2D images,
        :return: The reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D
        """
        if self.join_denoising:
            den_a_filter = SpitfireJoinDenoise(weight=self.weighting_inner, reg=self.reg_inner)
            a = 32*den_a_filter(reg_image).detach()
            den_b_filter = SpitfireJoinDenoise(weight=self.weighting_outer, reg=self.reg_outer)
            b = 32*den_b_filter(image).detach()
        else:
            a = torch.sum(reg_image, axis=0)
            b = torch.sum(image, axis=0)
            den_a_filter = SpitfireDenoise(weight=self.weighting_inner, reg=self.reg_inner)
            a = den_a_filter(a).detach()
            den_b_filter = SpitfireDenoise(weight=self.weighting_outer, reg=self.reg_outer)
            b = den_b_filter(b).detach()

        if self.epsilon == 'map':
            map_ = SureMap(smooth=False)
            epsilon = map_(image[0, ...], a, b)
            print('epsilon map=', epsilon.shape)
        elif self.epsilon == 'mode':
            map_ = SureMap(smooth=False)
            epsilon = map_.mode(image[0, ...], a, b)
            print('mode epsilon=', epsilon)
        else:
            epsilon = float(self.epsilon)
        out = a - epsilon * b
        return self._crop(torch.nn.functional.relu(out, inplace=True))


metadata = {
    'name': 'ISFEDDenoising',
    'label': 'ISFED Denoising',
    'class': ISFEDDenoising,
    'parameters': {
        'epsilon': {
            'type': str,
            'label': 'epsilon',
            'help': 'Weighting parameter',
            'default': 0.3
        },
        'reg_inner': {
            'type': float,
            'label': 'Inner reg',
            'help': 'Regularisation for inner detectors denoising',
            'default': 0.995
        },
        'reg_outer': {
            'type': float,
            'label': 'Outer reg',
            'help': 'Regularisation for outer detectors denoising',
            'default': 0.995
        },
        'weighting': {
            'type': float,
            'label': 'Sparsity weighting',
            'help': 'Denoising sparsity: 0.1 sparse, 0.6 moderately sparse, 0.9 not sparse',
            'default': 0.9,
            'range': (0, 1)
        }
    }
}
