from types import SimpleNamespace
from pprint import pprint
import logging
from typing import Callable
import gin
from functools import partial
from tu.configs import nested_dict_to_dot_map_dict
logger = logging.getLogger(__name__)


class RuntimeConfig(SimpleNamespace):
    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"Cannot reassign value to '{name}'.")
        super().__setattr__(name, value)

    def _to_nested_dict(self):
        return {
            k: str(v) if not isinstance(v, RuntimeConfig) else v._to_nested_dict()
            for k, v in self.__dict__.items()}

    def update_gin_configs(self):
        dd = nested_dict_to_dot_map_dict(self._to_nested_dict())

        bindings = [f"{k} = {str(v)}" for k, v in dd.items()]
        pprint(bindings)
        gin.config.parse_config(bindings)
