from type_utils import P, T

__all__ = ['translation_matrix', 'rotation_matrix', 'scale_matrix', 'reflection_matrix', 'identity_matrix']

def rotation_matrix(angle: float, direction: P, point: P) -> T:
    """
    Args:
        angle (float) : the angle of rotation in radians
        direction (P) : the axis of rotation
        point (P) : the point about which the rotation is performed
    """
def translation_matrix(offset: P) -> T:
    """
    Args:
        offset (P) : the translation vector
    """
def scale_matrix(scale: float, origin: P) -> T:
    """
    Args:
        scale (float) - the scaling factor, only uniform scaling is supported
        origin (P) - the origin of the scaling operation
    """
def reflection_matrix(point: P, normal: P) -> T:
    """
    Args:
        point: P - a point on the mirror plane
        normal: P - the normal vector of the mirror plane
    """
def identity_matrix() -> T:
    """
    Returns the identity matrix in SE(3).
    """
