from .interface import SAiryscanEnhancing
import torchvision.transforms as transforms


class SAiryscanGaussian(SAiryscanEnhancing):
    """Apply a gaussian filter

    Parameters
    ----------
    sigma: float or list
        Gaussian sigma in each dimension

    """
    def __init__(self, sigma=0.5, kernel_size=(11, 11)):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size

    def __call__(self, image):
        out_bc = image.view(1, 1, image.shape[0], image.shape[1])
        out_b = transforms.functional.gaussian_blur(out_bc,
                                                    kernel_size=self.kernel_size,
                                                    sigma=self.sigma)
        return out_b.view(image.shape)
