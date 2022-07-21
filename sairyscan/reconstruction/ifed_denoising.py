import torch
from .interface import SAiryscanReconstruction
from ._sure import SureMap
from .spitfire_join_denoise import SpitfireJoinDenoise
from sairyscan.enhancing.spitfire_denoise import SpitfireDenoise


class IFEDDenoising(SAiryscanReconstruction):
    """Reconstruct a high resolution image with a ISFED method including a denoising step

    The denoising is performed on the two terms of the ISFED image difference using the SPITFIR(e)
    algorithm

    Parameters
    ----------
    epsilon: str or float
        weighting parameter for the ISFED difference second term. If epsilon='map', epsilon is an
        automatic estimated weight map using the SURE criterion. If epsilon='mode', epsilon is a
        float which corresponds to the main mode of the SURE map. Otherwise, epsilon can be fixed
        to any float value
    reg_inner: float
        Regularization for denoising of the first term of ISFED reconstruction
    reg_outer: float
        Regularization for the denoising of the second term of the SFED reconstruction
    weighting: float
        Weighting parameter of the SPITFIR(e) denoising model. Must be in [0, 1], with value close
        to 0 for sparse signal and close to one otherwise.

    """
    def __init__(self, inner_ring_index=7, epsilon=0.3, reg_inner=0.995,
                 reg_outer=0.995, weighting=0.9, join_denoising=False):
        super().__init__()

        print('ifed constructor inner index=', inner_ring_index)
        print('ifed constructor epsilon=', epsilon)
        self.num_args = 1
        self.reg_inner = reg_inner
        self.reg_outer = reg_outer
        self.inner_ring_index = inner_ring_index
        self.epsilon = epsilon
        self.weighting = weighting
        self.join_denoising = join_denoising
        self.map_ = None
        if inner_ring_index not in [7, 19]:
            raise ValueError('Inner ring index must be in (7, 19)')

    def __call__(self, image):
        """Reconstruct the IFED image from raw airyscan data

        Parameters
        ----------
        image: Tensor
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        self.progress(0)
        self.notify('IFED: sum inner and outer detectors')
        if self.join_denoising:
            den_a_filter = SpitfireJoinDenoise(weight=self.weighting, reg=self.reg_inner)
            a = den_a_filter(image[0:self.inner_ring_index, ...]).detach()
            den_b_filter = SpitfireJoinDenoise(weight=self.weighting, reg=self.reg_outer)
            b = den_b_filter(image[self.inner_ring_index + 1:32, ...]).detach()

            a *= self.inner_ring_index
            b *= (32-self.inner_ring_index)
        else:
            a = torch.sum(image[0:self.inner_ring_index, ...], axis=0)
            b = torch.sum(image[self.inner_ring_index + 1:32, ...], axis=0)
            self.notify('IFED: denoise rings')
            den_a_filter = SpitfireDenoise(weight=self.weighting, reg=self.reg_inner)
            a = den_a_filter(a)
            den_b_filter = SpitfireDenoise(weight=self.weighting, reg=self.reg_outer)
            b = den_b_filter(b)

        self.progress(33)
        print('ifed epsilon=', self.epsilon)
        if self.epsilon == 'map':
            self.notify('IFED: compute epsilon map')
            map_ = SureMap(smooth=False)
            epsilon = map_(image[0, ...], a, b)
            self.map_ = epsilon
            print('epsilon map=', epsilon.shape)
        elif self.epsilon == 'mode':
            self.notify('IFED: estimate epsilon using sure')
            map_ = SureMap(smooth=False)
            epsilon = map_.mode(image[0, ...], a, b)
            self.map_ = epsilon
            print('mode epsilon=', epsilon)
        else:
            self.notify('IFED: use manual epsilon')
            epsilon = float(self.epsilon)
        self.progress(75)
        self.notify('IFED: do sum and relu')
        out = a - epsilon * b
        self.progress(100)
        return self._crop(torch.nn.functional.relu(out, inplace=True))


metadata = {
    'name': 'IFEDDenoising',
    'label': 'IFED Denoising',
    'class': IFEDDenoising,
    'parameters': {
        'inner_ring_index': {
            'type': int,
            'label': 'Inner index',
            'help': 'Index of the inner ring last detector (7, 19)',
            'default': 7
        },
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
