import logging
import numpy as np
from scipy.spatial.transform import Rotation
from transforms3d._gohlketransforms import (
    translation_matrix as _translation_matrix,
    rotation_matrix as _rotation_matrix, reflection_matrix as _reflection_matrix, identity_matrix as _identity_matrix
)
from type_utils import P, T
from typing import Union
from engine.constants import ENGINE_MODE

logger = logging.getLogger(__name__)


__all__ = [
    "translation_matrix",
    "identity_matrix"
] 


def translation_matrix(offset: P) -> T:
    """
    Args:
        offset (P) : the translation vector, which must be composed of integers
    """
    return _translation_matrix(offset)


def identity_matrix() -> T:
    """
    Returns the identity matrix in SE(3).
    """
    return _identity_matrix()
