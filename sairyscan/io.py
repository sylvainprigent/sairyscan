"""Airyscan IO

Set of IO methods to read Airyscan raw files

Classes
-------
AiryscanReader

"""
from pathlib import Path
import numpy as np
import torch
from aicspylibczi import CziFile
from sairyscan.settings import Settings


class AiryscanReader:
    """Read a CZI Zeiss raw data

    Parameters
    ----------
    filename: str
        Path to the .czi file

    """
    def __init__(self, filename):
        super().__init__()
        if filename.endswith('.czi') or filename.endswith('.CZI'):
            self.filename = filename
        else:
            raise IOError('AiryscanReader can only read CZI files')

        pth = Path(filename)
        self.czi = CziFile(pth)
        self.dimensions = self.czi.get_dims_shape()[0]

    def frames(self):
        """Get the number of frames in the CZI file"""
        return self.dimensions['S'][1]

    def depth(self):
        """Get the number of slices in the CZI file"""
        return self.dimensions['Z'][1]

    def frame(self, f=0, c=0):
        """Extract one frame from the CZI data

        Parameters
        ----------
        f: int
            Frame index
        c: int
            Channel index

        Returns
        -------
        a numpy array [H, Z, Y, X] for 3D slice, [H, Y, X] for 2D slices. Where H is the detector
        dimension

        """
        img, shp = self.czi.read_image(S=f, C=c)
        return torch.from_numpy(np.float32(np.squeeze(img))).to(Settings.instance().device)
