import torch
from sairyscan.enhancing.interface import SAiryscanEnhancing


def hv_loss(img, weighting):
    """Sparse Hessian regularization term

    Parameters
    ----------
    img: Tensor
        Tensor of shape BCYX containing the estimated image
    weighting: float
        Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse

    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    hv = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                    (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1])).sum()
    return hv / (a * b * h * w)


def dataterm_denoise(noisy_image, denoised_image):
    """Denoising L2 data-term

    Compute the L2 error between the original image and the convoluted reconstructed image

    Parameters
    ----------
    noisy_image: Tensor
        Tensor of shape BCZYX containing the original blurry image
    denoised_image: Tensor
        Tensor of shape BCYX containing the estimated deblurred image

    """
    mse_ = 0
    for i in range(noisy_image.shape[2]):
        mse_ += torch.sum(torch.square(noisy_image[:, :, i, :, :] - denoised_image))
    return mse_ / noisy_image.numel()


class SpitfireJoinDenoise(SAiryscanEnhancing):
    """Gray scaled image deconvolution with the Spitfire algorithm

    Parameters
    ----------
    weight: float
        model weight between hessian and sparsity. Value is in  ]0, 1[
    reg: float
        Regularization weight. Value is in [0, 1]

    """

    def __init__(self, weight=0.6, reg=0.5):
        super().__init__()
        self.weight = weight
        self.reg = reg
        self.niter_ = 0
        self.max_iter_ = 2000
        self.gradient_step_ = 0.01
        self.loss_ = None

    def __call__(self, image):
        if image.ndim == 3:
            return self.run_2d(image)
        elif image.ndim == 4:
            raise NotImplementedError('Spitfire 3D deconvolution is not yet implemented')

    def run_2d(self, image):
        self.progress(0)
        mini = torch.min(image)
        maxi = torch.max(image)
        image = (image-mini)/(maxi-mini)
        # pad image
        #padding = 13
        #pad_fn = torch.nn.ReflectionPad2d(padding)
        #image_pad = pad_fn(image.detach().clone().view(1, 1, image.shape[0], image.shape[1]))
        image_pad = image.view(1, 1, image.shape[0], image.shape[1], image.shape[2])

        deconv_image = torch.mean(image_pad, axis=2)
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0
        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_denoise(image_pad, deconv_image) + \
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
        out = deconv_image.view(image_pad.shape[3], image_pad.shape[4])
        return (maxi-mini)*out+mini

metadata = {
    'name': 'SpitfireJoinDenoise',
    'label': 'Spitfire Join Denoise',
    'class': SpitfireJoinDenoise,
    'parameters': {
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
