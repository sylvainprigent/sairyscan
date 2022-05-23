from .interface import SAiryscanEnhancing
import torchvision.transforms as transforms


class SAiryscanGaussian(SAiryscanEnhancing):
    """Apply a gaussian filter

    Parameters
    ----------
    sigma: float or list
        Gaussian sigma in each dimension

    """
    def __init__(self, sigma=0.5, kernel_size=7):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size

    def __call__(self, image):
        out_bc = image.view(1, 1, image.shape[0], image.shape[1])
        out_b = transforms.functional.gaussian_blur(out_bc,
                                                    kernel_size=(self.kernel_size,
                                                                 self.kernel_size),
                                                    sigma=self.sigma)
        return out_b.view(image.shape)


metadata = {
    'name': 'SAiryscanGaussian',
    'label': 'Gaussian Filter',
    'class': SAiryscanGaussian,
    'parameters': {
        'sigma': {
            'type': float,
            'label': 'Sigma',
            'help': 'Gaussian sigma parameter',
            'default': 0.5,
            'range': (0, 512)
        },
        'kernel_size': {
            'type': int,
            'label': 'kernel size',
            'help': 'Size of the kernel in the x and y directions',
            'default': 7,
            'range': (3, 512)
        }
    }
}
