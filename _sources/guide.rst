Guide
=====

Design
------

A `SAiryscan` reconstruction pipeline as porposed in the Optics Letters paper is made of 3 consecutive steps:

1. **Registration**: is a module to spatially co-register all the 32 detectors to a reference detector (usually the central one). This step is optional.

2. **Reconstruction**: is a module to transform an image stack of 32 detectors into a single gray scaled image. This is the only mandatory step. 

3. **Enhancing**: is a module to enhance the quality of the reconstructed image with denoising or deconvolution for example. This step is optional.


Run a reconstruction with the API
---------------------------------

This section shows a short introduction of how to use the `SAiryscan` library. Please refer to
the API :doc:`/modules` documentation for more advanced features.

The pipeline API is :class:`SAiryscanPipeline <sairyscan.api.SAiryscanPipeline>`. It allows to create a pipeline 
from reconstruction modules. Bellow an example on how to create a pipeline:

.. code-block:: python

    from sairyscan.registration import SRegisterPosition
    from sairyscan.reconstruction import ISM
    from sairyscan.enhancing import SAiryscanWiener
    from sairyscan.enhancing import PSFGaussian
    from sairyscan.api import SAiryscanPipeline

    from sairyscan.data import celegans

    registration = SRegisterPosition()
    reconstruction = ISM()
    psf = PSFGaussian(sigma=(1.5, 1.5), shape=(7, 7))
    enhancing = SAiryscanWiener(psf, beta = 1e-5)
    pipeline = SAiryscanPipeline(registration, reconstruction, enhancing)

    image = celegans()
    reconstructed_image = pipeline(image)


Build custom reconstruction modules
-----------------------------------

Each of the 3 modules are based on interfaces :class:`SAiryscanRegistration <sairyscan.registration.SAiryscanRegistration>`, 
:class:`SAiryscanReconstruction <sairyscan.reconstruction.SAiryscanReconstruction>` and :class:`SAiryscanEnhancing <sairyscan.enhancing.SAiryscanEnhancing>`. 

Thus, to implement a new module, we just need to implement one of the interface.


Create a custom registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bellow an example of custom registration code structure:

.. code-block:: python

    from sairyscan.registration import SAiryscanRegistration

    class MyCustomRegistration(SAiryscanRegistration):
        '''A custom registration module
        
        All the module settings must be set to the constructor

        :param param1: One setting
        :param param2: Another setting

        '''
        def __init__(self, param1: float, param2: float):
            super().__init__()
            self.__param1 = param1
            self.__param2 = param2

        def __call__(self, image: torch.Tensor) -> torch.Tensor:
            '''The implementation is done in the call method
            
            :param image: Raw airyscan data for a single channel time point [H (Z) Y X]
            :return: Co-registered detectors [H (Z) Y X]

            '''
            # This is a fake registration that does nothing
            return image


Create a custom reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bellow an example of custom reconstruction code structure:

.. code-block:: python

    from sairyscan.reconstruction import SAiryscanReconstruction

    class MyCustomReconstruction(SAiryscanReconstruction):
        '''A custom reconstruction module
        
        All the module settings must be set to the constructor

        :param param1: One setting
        :param param2: Another setting

        '''
        def __init__(self, param1: float, param2: float):
            super().__init__()
            self.__param1 = param1
            self.__param2 = param2

        def __call__(self, image: torch.Tensor, reg_image: torch.Tensor) -> torch.Tensor:
            """Do the reconstruction

            :param image: Raw detector stack to reconstruct [H (Z) Y X]
            :param reg_image: Spatially co-registered detectors stack [H (Z) Y X]
            :return: High resolution image [(Z) Y X]
            """
            # This implementation is similar to confocal image with 1.25 pinholes
            return torch.sum(image)


Create a custom enhancing
~~~~~~~~~~~~~~~~~~~~~~~~~

Bellow an example of custom enhancing module code structure:

.. code-block:: python

    from sairyscan.enhancing import SAiryscanEnhancing

    class MyCustomEnhancing(SAiryscanEnhancing):
        '''A custom enhancing module
        
        All the module settings must be set to the constructor

        :param param1: One setting
        :param param2: Another setting

        '''
        def __init__(self, param1: float, param2: float):
            super().__init__()
            self.__param1 = param1
            self.__param2 = param2

        def __call__(self, image: torch.Tensor) -> torch.Tensor:
            """Do the enhancing

            :param image: Image to enhance [Y, X] or [Z, Y, X]
            :return: The enhanced image
            """
            # This example does nothing
            return image
