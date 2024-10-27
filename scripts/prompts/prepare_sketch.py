from pathlib import Path
import numpy as np

from helper import *
import mitsuba as mi
import traceback
import ipdb

import random
import math
import sys
import os

import mi_helper  # such that primitive call will be implemented

from dsl_utils import library, _children


def primitive_call(name, *args, **kwargs):
    return []


def library_call(func_name, *args, **kwargs):
    from dsl_utils import FAKE_CALL
    if FAKE_CALL:
        _children.add(func_name)
        return []
    if func_name not in library:
        for alt_func_name in library.keys():
            if library[alt_func_name]['docstring'] == func_name:
                # print(f'WARNING: {func_name=} not found in library but found an alternative: {alt_func_name=}')
                return library[alt_func_name]['__target__']()
        for alt_func_name in library.keys():
            if library[alt_func_name]['docstring'].split(';')[0] == func_name:
                return library[alt_func_name]['__target__'](**kwargs)
        print(f'WARNING: {func_name=} not found in library')
        return []
    return library[func_name]['__target__']()


def loop(n: int, fn: Callable[[int], Shape]) -> Shape:
    return fn(0)
