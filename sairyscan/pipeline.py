import torch
from .io import SAiryscanReader


class SAiryscanLoop:
    """Apply a SAiryscan reconstruction method on all the frames and channels

    """
    def __init__(self, filename, to_file=False, destination_filename=''):
        self.filename = filename
        self.to_file = to_file
        self.destination_filename = destination_filename
        self.reader = SAiryscanReader(filename)

    def __call__(self, method):
        """Apply the reconstruction loop

        Parameters
        ----------
        method: SAiryscanFilter
            instance of airyscan reconstruction filter
        **kwargs: dict
            dictionary of the reconstruction method parameters

        Returns
        -------
        Tensor if the input to_file is false

        """
        if self.to_file:
            return self.reconstruct_to_file(method)
        else:
            return self.reconstruct_to_tensor(method)

    def reconstruct_to_tensor(self, method):
        if self.reader.depth() > 1:
            out_data = torch.zeros((self.reader.frames(), self.reader.channels(),
                                    self.reader.depth(),
                                    self.reader.height(), self.reader.width()))
        else:
            out_data = torch.zeros((self.reader.frames(), self.reader.channels(),
                                    self.reader.height(), self.reader.width()))
        for frame_idx in self.reader.frames():
            for channel_idx in self.reader.channels():
                raw_data = self.reader.data(frame_idx, channel_idx)
                out_data[frame_idx, channel_idx, ...] = method(raw_data)
        return out_data

    def reconstruct_to_file(self, method):
        # if frame=channels=1 save to file
        # elif frame=1 and channels>1 save one file per channel filename_c1.tif, filename_c2.tif
        # else save to multiple files:
        # - filename=directory
        # - one main file per channel: filename_c1.txt and sub files filename_c1_txx.tif
        raise NotImplementedError('SAiryscanLoop: reconstruct_to_file() not yet implemented')
