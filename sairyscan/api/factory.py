"""This module implements factory to instantiate models"""
import numpy as np
import torch

from ..core import SObservable


class SAiryscanFactoryError(Exception):
    """Raised when an error happen when a module is built in the factory"""


class SAiryscanModuleFactory:
    """Factory for SAiryscan modules"""
    def __init__(self):
        self._data = {}

    def register(self, key: str, metadata: dict[str, any]):
        """Register a new builder to the factory

        :param key: Name of the module to register,
        :param metadata: Dictionary containing the filter metadata
        """
        self._data[key] = metadata

    def get_parameters(self, key: str) -> dict[str, any]:
        """Get the parameters of a filter

        :param key: Name of the filter
        :return: The dictionary of parameters
        """
        return self._data[key]['parameters']

    def get_keys(self) -> list[str]:
        """Get the names of all the registered modules

        :return: The list of all the registered modules names
        """
        return list(self._data.keys())

    def get(self, key: str, **kwargs: dict[str, any]) -> SObservable:
        """Get the instance of the SAiryscan module

        :param key: Name of the module to load
        :param kwargs: Dictionary of CLI args for models parameters (ex: number of channels)
        :return: The module instance
        """
        metadata = self._data.get(key)
        if not metadata:
            raise ValueError(key)
        builder = SAiryscanModuleBuilder()
        return builder.get_instance(metadata, kwargs)


class SAiryscanModuleBuilder:
    """Interface for a SAiryscan module builder

    The builder is used by the factory to instantiate a module

    """
    def __init__(self):
        self._instance = None

    def get_instance(self, metadata: dict[str, any], args: dict[str, any]):
        """Get the instance of the module

        :param metadata: Metadata of the module
        :param args: Arguments to instantiate the module
        :return: instance of the module

        """
        # check the args
        instance_args = {}
        for key, value in metadata['parameters'].items():
            val = self._get_arg(value, key, args)
            instance_args[key] = val
        return metadata['class'](**instance_args)

    def _get_arg(self,
                 param_metadata: dict[str, any],
                 key: str,
                 args: dict[str, any]
                 ) -> int | float | str | bool | torch.Tensor:
        """Check and convert an argument value from parameters dictionary

        :param param_metadata: Information on the parameter
        :param key: Name of the parameter
        :param args: Value of the parameter
        :return: The value of the parameter in the proper type and format
        """
        type_ = param_metadata['type']
        print('param metadata:', param_metadata)
        range_ = None
        if 'range' in param_metadata:
            range_ = param_metadata['range']

        if type_ is float:
            return self.get_arg_float(args, key, param_metadata['default'],
                                      range_)
        if type_ is int:
            return self.get_arg_int(args, key, param_metadata['default'],
                                    range_)
        if type_ is bool:
            return self.get_arg_bool(args, key, param_metadata['default'],
                                     range_)
        if type_ is str:
            return self.get_arg_str(args, key, param_metadata['default'])
        if type_ is torch.Tensor:
            return self.get_arg_array(args, key, param_metadata['default'])
        if type_ == 'select':
            return self.get_arg_select(args, key, param_metadata['values'])
        raise ValueError(f"parameter type '{type_}' not recognized")

    @staticmethod
    def _error_message(key: str, value_type: str, value_range: tuple | None) -> str:
        """Build the error message to raise if an input parameter is not correct

        :param key: Input parameter key
        :param value_type: String naming the input type (int, float...)
        :param value_range: Min and max values of the parameter
        :return: The error message
        """
        range_message = ''
        if value_range and len(value_range) == 2:
            range_message = f' in range [{str(value_range[0]), str(value_range[1])}]'

        message = f'Parameter {key} must be of type `{value_type}` {range_message}'
        return message

    def get_arg_int(self,
                    args: dict[str, any],
                    key: str,
                    default_value: int,
                    value_range: tuple[int, ...] = None
                    ) -> int:
        """Get the integer value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args.

        :param args: Dictionary of the input args,
        :param key: Name of the parameters,
        :param default_value: Default value of the parameter,
        :param value_range: Min and max value of the parameter,
        :return: The parameter value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # cast
            try:
                value = int(args[key])
            except ValueError as err:
                raise SAiryscanFactoryError(self._error_message(key, 'int', value_range)) from err
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SAiryscanFactoryError(self._error_message(key, 'int', value_range))
        return value

    def get_arg_float(self,
                      args: dict[str, any],
                      key: str,
                      default_value: float,
                      value_range: tuple[float, ...] = None
                      ) -> float:
        """Get the float value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args.

        :param args: Dictionary of the input args,
        :param key: Name of the parameters,
        :param default_value: Default value of the parameter,
        :param value_range: Min and max value of the parameter,
        :return: The parameter value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # cast
            try:
                value = float(args[key])
            except ValueError as err:
                raise SAiryscanFactoryError(self._error_message(key, 'float', value_range)) from err
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SAiryscanFactoryError(self._error_message(key, 'float', value_range))
        return value

    def get_arg_str(self,
                    args: dict[str, any],
                    key: str,
                    default_value: str,
                    value_range: tuple[str, ...] = None):
        """Get the string value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args.

        :param args: Dictionary of the input args,
        :param key: Name of the parameters,
        :param default_value: Default value of the parameter,
        :param value_range: Min and max value of the parameter,
        :return The parameter value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # cast
            try:
                value = str(args[key])
            except ValueError as err:
                raise SAiryscanFactoryError(self._error_message(key, 'str', value_range)) from err
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SAiryscanFactoryError(self._error_message(key, 'str', value_range))
        return value

    def get_arg_bool(self,
                     args: dict[str, any],
                     key: str,
                     default_value: bool,
                     value_range: tuple[bool, ...] = None
                     ) -> bool:
        """Get the boolean value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args.

        :param args: Dictionary of the input args,
        :param key: Name of the parameters,
        :param default_value: Default value of the parameter,
        :param value_range: Min and max value of the parameter,
        :return: The parameter value
        """
        value = default_value
        # cast
        if isinstance(args, dict) and key in args:
            if isinstance(args[key], str):
                if args[key] == 'True':
                    value = True
                else:
                    value = False
            elif isinstance(args[key], bool):
                value = args[key]
            else:
                raise SAiryscanFactoryError(self._error_message(key, 'bool', value_range))
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SAiryscanFactoryError(self._error_message(key, 'bool', value_range))
        return value

    def get_arg_array(self,
                      args: dict[str, any],
                      key: str,
                      default_value: torch.Tensor
                      ) -> torch.Tensor:
        """Get the value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args.

        :param args: Dictionary of the input args,
        :param key: Name of the parameters,
        :param default_value: Default value of the parameter,
        :return: The parameter value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # rint('psf type=', type(args[key]))
            if isinstance(args[key], torch.Tensor):
                value = args[key]
            elif isinstance(args[key], np.ndarray):
                value = torch.Tensor(args[key])
            else:
                raise SAiryscanFactoryError(self._error_message(key, 'array', None))
        return value

    def get_arg_select(self,
                       args: dict[str, any],
                       key: str,
                       values: list[str]
                       ) -> str:
        """Get the select value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args.

        :param args: Dictionary of the input args,
        :param key: Name of the parameters,
        :param values: Default value of the parameter,
        :return: The parameter value
        """
        if isinstance(args, dict) and key in args:
            value = str(args[key])
            for x in values:
                if str(x) == value:
                    return x
        raise SAiryscanFactoryError(self._error_message(key, 'select', None))
