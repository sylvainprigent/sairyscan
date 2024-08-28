Modules
=======

API
---

.. currentmodule:: sairyscan.api

.. autosummary::
    :toctree: generated
    :nosignatures:

    SAiryscanAPI
    SAiryscanPipeline
    SAiryscanLoop


Registration
------------

.. currentmodule:: sairyscan.registration

.. autosummary::
    :toctree: generated
    :nosignatures:

    SAiryscanRegistration
    SRegisterPosition
    SRegisterFourierPhase
    SRegisterMSE


Reconstruction
--------------

.. currentmodule:: sairyscan.reconstruction

.. autosummary::
    :toctree: generated
    :nosignatures:

    SAiryscanReconstruction
    IFED
    ISFED
    ISM
    PseudoConfocal
    SpitfireJoinDeconv
    SpitfireJoinDenoise
    SSpitfireFlow
    SureMap
    IFEDDenoising
    ISFEDDenoising


Enhancing
---------

.. currentmodule:: sairyscan.enhancing

.. autosummary::
    :toctree: generated
    :nosignatures:

    SAiryscanEnhancing
    SAiryscanGaussian
    SAiryscanWiener
    SAiryscanRichardsonLucy
    SpitfireDeconv


Example data
------------

.. currentmodule:: sairyscan.data

.. autosummary::
    :toctree: generated
    :nosignatures:

    celegans

Internal patterns
-----------------

Interfaces are the core of the framework to define modules that store metadata and allows to
efficiently build training and prediction workflows.

The available interfaces are:

.. currentmodule:: sairyscan.core

.. autosummary::
    :toctree: generated
    :nosignatures:

    SAiryscanReader
    Settings
    SettingsContainer
    SObservable
    SObserver
    SObserverConsole
