"""This module implement a Spitfire deconvolution"""
import torch
from .interface import SAiryscanEnhancing


def hv_loss(img: torch.Tensor, weighting: float) -> torch.Tensor:
    """Sparse Hessian regularization term

    :param img: Tensor of shape BCYX containing the estimated image
    :param weighting: Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse
    :return: The loss value
    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    hv = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                    (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1])).sum()
    return hv / (a * b * h * w)


def dataterm_deconv(blurry_image: torch.Tensor, deblurred_image: torch.Tensor, psf: torch.Tensor):
    """Deconvolution L2 data-term

    Compute the L2 error between the original image and the convoluted reconstructed image.

    :param blurry_image: Tensor of shape BCYX containing the original blurry image,
    :param deblurred_image: Tensor of shape BCYX containing the estimated deblurred image,
    :param psf: Tensor containing the point spread function,
    :return: The data term value
    """
    conv_op = torch.nn.Conv2d(1, 1, kernel_size=psf.shape[2],
                              stride=1,
                              padding=int((psf.shape[2] - 1) / 2),
                              bias=False)
    with torch.no_grad():
        conv_op.weight = torch.nn.Parameter(psf)
    mse = torch.nn.MSELoss()
    return mse(blurry_image, conv_op(deblurred_image))


class SpitfireDeconv(SAiryscanEnhancing):
    """Gray scaled image deconvolution with the Spitfire algorithm

    :param psf: Point spread function,
    :param weight: Model weight between hessian and sparsity. Value is in  ]0, 1[,
    :param reg: Regularization weight. Value is in [0, 1]
    """
    def __init__(self, psf: torch.Tensor, weight: float = 0.6, reg: float = 0.995):
        super().__init__()
        self.psf = psf
        self.weight = weight
        self.reg = reg
        self.niter_ = 0
        self.max_iter_ = 2000
        self.gradient_step_ = 0.01
        self.loss_ = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the deconvolution

        :param image: Image to deblur
        :return: The deblurred image
        """
        if image.ndim == 2:
            return self.run_2d(image)
        raise NotImplementedError('Spitfire 3D deconvolution is not yet implemented')

    def run_2d(self,  image: torch.Tensor) -> torch.Tensor:
        """Do the deconvolution for the 2D case

        :param image: Image to deblur
        :return: The deblurred image
        """
        self.progress(0)
        image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
        # pad image
        padding = 13
        pad_fn = torch.nn.ReflectionPad2d(padding)
        image_pad = pad_fn(image.detach().clone().view(1, 1, image.shape[0], image.shape[1]))

        self.psf = self.psf.view(1, 1, self.psf.shape[0], self.psf.shape[1])
        deconv_image = image_pad.detach().clone()
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0
        loss = None
        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_deconv(image_pad, deconv_image, self.psf) + \
                (1-self.reg) * hv_loss(deconv_image, self.weight)
            print('iter:', self.niter_, ' loss:', loss.item())
            if abs(loss - previous_loss) < 1e-7:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss
        self.progress(100)
        return deconv_image.view(image_pad.shape[2],
                                 image_pad.shape[3])[padding:-padding, padding:-padding]


metadata = {
    'name': 'SpitfireDeconv',
    'label': 'Spitfire Deconv',
    'class': SpitfireDeconv,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'weight': {
            'type': float,
            'label': 'weight',
            'help': 'Model weight between hessian and sparsity. Value is in  ]0, 1[',
            'default': 0.6,
            'range': (0, 1)
        },
        'reg': {
            'type': float,
            'label': 'Regularization',
            'help': 'Regularization weight. Value is in [0, 1]',
            'default': 0.995,
            'range': (0, 1)
        }
    }
}
