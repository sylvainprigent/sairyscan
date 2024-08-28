"""This module implement the SAiryscan main API"""
import os
import importlib

from ..core import SAiryscanReader, SObservable
from ..registration import SAiryscanRegistration
from ..reconstruction import SAiryscanReconstruction
from ..enhancing import SAiryscanEnhancing

from .factory import SAiryscanModuleFactory
from .pipeline import SAiryscanPipeline, SAiryscanLoop


class SAiryscanAPI:
    """Main API to build an Airyscan data reconstruction pipeline"""
    def __init__(self):
        self.filters = SAiryscanModuleFactory()
        discovered_modules = self._find_modules()
        for name in discovered_modules:
            # print('register the module:', name)
            mod = importlib.import_module(name)
            # print(mod.__name__)
            self.filters.register(mod.metadata['name'], mod.metadata)

    @staticmethod
    def _find_modules() -> list[str]:
        """Search for modules to instantiate in the factory

        :return: The list of module found
        """
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.dirname(path)
        modules = []
        for parent in ['enhancing', 'reconstruction', 'registration']:
            path_ = os.path.join(path, parent)
            for x in os.listdir(path_):
                if (x.endswith(".py") and 'interface' not in x
                        and '__init__' not in x and not x.startswith("_")):
                    modules.append(f"sairyscan.{parent}.{x.split('.')[0]}")
        return modules

    def filter(self, name: str, **args) -> SObservable | None:
        """Get a filter

        :param name: Unique name of the filter
        :return: An instance of the filter if it exists, None otherwise
        """
        if name == 'None':
            return None
        return self.filters.get(name, **args)

    @staticmethod
    def reader(filename: str) -> SAiryscanReader:
        """Get the airyscan image file reader

        :param filename: Path of the image file
        :return: An intance of the reader
        """
        return SAiryscanReader(filename)

    @staticmethod
    def pipeline(reconstruction: SAiryscanReconstruction,
                 registration: SAiryscanRegistration = None,
                 enhancing: SAiryscanEnhancing = None
                 ) -> SAiryscanPipeline:
        """Get a reconstruction pipeline

        :param reconstruction: instance of the reconstruction filter
        :param registration: instance of the registration filter
        :param enhancing: instance of the enhancing filter
        :return: An instance of the pipeline
        """
        return SAiryscanPipeline(reconstruction, registration, enhancing)

    @staticmethod
    def loop(method: SObservable,
             filename: str,
             to_file: bool = False,
             destination_filename: str = ''
             ) -> SAiryscanLoop:
        """Get an instance of pipeline loop

        :param method: Pipeline or filter to apply,
        :param filename: Path to the file containing the data,
        :param to_file: True to dave the tensor to file. Otherwise, the call method return a tensor
        :param destination_filename: Path to the file where the result is saved
        :return: An instance of the loop
        """
        instance_ = SAiryscanLoop(filename, to_file, destination_filename)
        return instance_(method)
