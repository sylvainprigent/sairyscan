import os
from sairyscan.enhancing import Spitfire, PSFGaussian
import numpy as np
import torch
from skimage.io import imread
import matplotlib.pyplot as plt

from sairyscan.api import SAiryscanAPI

my_test_file = '/Users/sprigent/Documents/code/github/sairyscan/sairyscan/enhancing/tests/ism_celegans.tif'


image_numpy = np.float32(imread(my_test_file))
image = torch.Tensor(image_numpy/np.max(image_numpy))
#image = torch.Tensor(image_numpy)

psf_generator = PSFGaussian((1.6, 1.6), (11, 11))
psf = psf_generator()

filter = Spitfire(psf, weight=0.5, reg=0.995)
out_image = filter(image)

print('out image shape=', out_image.shape)

plt.figure(figsize=(10, 10))
plt.title('psf')
plt.imshow(psf.numpy(), cmap='gray')
plt.axis('off')

plt.figure(figsize=(10, 10))
plt.title('Original')
plt.imshow(image.numpy(), cmap='gray')
plt.axis('off')

plt.figure(figsize=(10, 10))
plt.title('Spitire deconv')
plt.imshow(out_image.detach().numpy(), cmap='gray')
plt.axis('off')
plt.show()
