Install
=======

This section contains the instructions to install ``SAiryscan``

Using pip from PyPI
-------------------

Releases are available in PyPI. We recommend using virtual environment.
Depending on the GPU and ``PyTorch`` version you are using you may need to install various packages.
For default local usage:

.. code-block:: shell

    python -m venv .venv
    source .env/bin/activate
    pip install sairyscan

Using pip from sources
----------------------

Releases are available in a GitHub repository. We recommend using virtual environment.
Depending on the GPU and ``PyTorch`` version you are using you may need to install various packages.
For default local usage:

.. code-block:: shell

    python -m venv .venv
    source .env/bin/activate
    pip install https://github.com/sylvainprigent/sairyscan/archive/master.zip


From source
-----------

If you plan to develop ``SAiryscan`` or want to install locally from sources

.. code-block:: shell

    python -m venv .venv
    source .venv/bin/activate
    git clone https://github.com/sylvainprigent/sairyscan.git
    cd sairyscan
    pip install -e . # or 'pip install -r requirements.txt' for dev dependencies
