import os
import importlib
from .factory import SAiryscanModuleFactory
from sairyscan.core import SAiryscanPipeline, SAiryscanLoop


class SAiryscanAPI:
    def __init__(self):
        # Filters factory
        self.filters = SAiryscanModuleFactory()
        discovered_modules = self._find_modules()
        for name in discovered_modules:
            print('register the module:', name)
            mod = importlib.import_module(name)
            print(mod.__name__)
            self.filters.register(mod.metadata['name'], mod.metadata)

    @staticmethod
    def _find_modules():
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.dirname(path)
        modules = []
        for parent in ['enhancing', 'reconstruction', 'registration']:
            path_ = os.path.join(path, parent)
            for x in os.listdir(path_):
                if x.endswith("wiener.py") and 'interface' not in x and '__init__' not in x:
                    modules.append(f"sairyscan.{parent}.{x.split('.')[0]}")
        return modules

    def filter(self, name, **args):
        return self.filters.get(name, **args)

    @staticmethod
    def pipeline(reconstruction, registration=None, enhancing=None):
        return SAiryscanPipeline(reconstruction, registration, enhancing)

    @staticmethod
    def loop(method, filename, to_file=False, destination_filename=''):
        instance_ = SAiryscanLoop(filename, to_file, destination_filename)
        return instance_(method)
