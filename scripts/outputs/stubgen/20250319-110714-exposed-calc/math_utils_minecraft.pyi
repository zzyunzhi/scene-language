from type_utils import P, T

__all__ = ['translation_matrix', 'identity_matrix']

def translation_matrix(offset: P) -> T:
    """
    Args:
        offset (P) : the translation vector, which must be composed of integers
    """
def identity_matrix() -> T:
    """
    Returns the identity matrix in SE(3).
    """
