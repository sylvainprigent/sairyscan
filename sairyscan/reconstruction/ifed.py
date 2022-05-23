import torch
from .interface import SAiryscanReconstruction
from ._sure import SureMap


class IFED(SAiryscanReconstruction):
    def __init__(self, inner_ring_index=7, epsilon=0.3):
        super().__init__()

        print('ifed constructor inner index=', inner_ring_index)
        print('ifed constructor epsilon=', epsilon)
        self.num_args = 1
        self.inner_ring_index = inner_ring_index
        self.epsilon = epsilon
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
        a = torch.sum(image[0:self.inner_ring_index, ...], axis=0)
        b = torch.sum(image[self.inner_ring_index + 1:32, ...], axis=0)
        self.progress(33)
        print('ifed epsilon=', self.epsilon)
        if self.epsilon == 'map':
            self.notify('IFED: compute epsilon map')
            map_ = SureMap(smooth=True)
            epsilon = map_(image[0, ...], a, b)
            print('epsilon map=', epsilon.shape)
        elif self.epsilon == 'mode':
            self.notify('IFED: estimate epsilon using sure')
            map_ = SureMap(smooth=False)
            epsilon = map_.mode(image[0, ...], a, b)
            print('mode epsilon=', epsilon)
        else:
            self.notify('IFED: use manual epsilon')
            epsilon = self.epsilon
        self.progress(75)
        self.notify('IFED: do sum and relu')
        out = a - epsilon * b
        self.progress(100)
        return torch.nn.functional.relu(out, inplace=True)


metadata = {
    'name': 'IFED',
    'label': 'IFED',
    'class': IFED,
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
        }
    }
}
