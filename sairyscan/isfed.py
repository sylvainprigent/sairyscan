import torch
import torchvision.transforms as transforms


class ISFED:
    def __init__(self, epsilon=0.3, smooth=False):
        self.epsilon = epsilon
        self.smooth = smooth

    def __call__(self, image, reg_image):
        """Reconstruct the ISFED image from raw airyscan data

        Parameters
        ----------
        image: ndarray
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images
        reg_image
            Co-registered airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        ndarray: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        out = torch.sum(reg_image, axis=0) - self.epsilon *torch.sum(image, axis=0)
        out = torch.nn.functional.relu(out, inplace=True)
        if self.smooth:
            out_bc = out.view(1, 1, out.shape[0], out.shape[1])
            out_b = transforms.functional.gaussian_blur(out_bc, kernel_size=(11, 11), sigma=0.5)
            return out_b.view(out.shape)
        else:
            return out
