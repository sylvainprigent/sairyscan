"""This module implements the generation of Gaussian PSFs"""
import math
import numpy as np
import torch


class PSFGaussian:
    """Generate a Gaussian PSF

    :param sigma: Radius of the Gaussian in each dimension
    :param shape: Size of the PSF array in each dimension
    """
    def __init__(self, sigma: tuple[int, ...], shape: tuple[int, ...]):
        self.sigma = sigma
        self.shape = shape
        self.psf_ = None

    def __call__(self) -> torch.Tensor:
        """Calculate the PSF image"""
        if len(self.shape) == 2:
            self.psf_ = np.zeros(self.shape)
            x0 = math.floor(self.shape[0] / 2)
            y0 = math.floor(self.shape[1] / 2)
            sigma_x2 = 0.5 / (self.sigma[0] * self.sigma[0])
            sigma_y2 = 0.5 / (self.sigma[1] * self.sigma[1])
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    self.psf_[x, y] = math.exp(- pow(x-x0, 2) * sigma_x2
                                               - pow(y-y0, 2) * sigma_y2)
        elif len(self.shape) == 3:
            x0 = self.shape[0] / 2
            y0 = self.shape[1] / 2
            z0 = self.shape[2] / 2
            sigma_x2 = 0.5 / self.sigma[0] * self.sigma[0]
            sigma_y2 = 0.5 / self.sigma[1] * self.sigma[1]
            sigma_z2 = 0.5 / self.sigma[2] * self.sigma[2]
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    for z in range(self.shape[2]):
                        self.psf_[z, x, y] = math.exp(- pow(x-x0, 2) * sigma_x2
                                                      - pow(y-y0, 2) * sigma_y2
                                                      - pow(z-z0, 2) * sigma_z2)
        else:
            raise ValueError('PSFGaussian: can generate only 2D or 3D PSFs')
        return torch.tensor(self.psf_/np.sum(self.psf_)).float()
