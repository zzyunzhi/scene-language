from type_utils import P, Shape

__all__ = ['assert_connect', 'assert_disconnect', 'assert_attach', 'assert_align_with_min', 'assert_align_with_center', 'assert_components', 'assert_bound', 'test_library_call']

def test_library_call(func_name: str, **kwargs) -> Shape:
    """
    A context manager; test a function from the library and yield its outputs.

    Args:
        func_name (str): Function name.
        **kwargs: Keyword arguments passed to the function.
    """
def assert_connect(shapes: list[str]):
    """
    Assert that all shapes have at least one intersecting 3D point.
    This is a stronger constraint than pairwise intersection.
    """
def assert_disconnect(shapes: list[str]):
    """
    Assert that any pair of shapes have no intersecting 3D points.
    """
def assert_bound(direction: P, bmin: float | str | None, bmax: float | str | None, shape: str):
    """
    Asserts that the shape's extent along the direction is bounded by 'bmin' and 'bmax'.
    If 'bmin' ('bmax') is a string, the bound is computed as the min (max) projection of ANY shape instance with this name.
    This function makes assertion for ALL shape instances with name `shape`.
    """
def assert_attach(direction: P, shapes: list[str]):
    """
    Asserts that each shape is attached sequentially to the next along the given direction.
    """
def assert_align_with_min(normal: P, shapes: list[str]):
    """
    Asserts that the 'minimum' points of the shapes, when projected to the direction of the normal vector, coincide at the same location.
    Note: Reversing the normal swaps 'minimum' with 'maximum'.
    Runs test for ANY combination of shape instances with the given name `shapes`.
    """
def assert_align_with_center(normal: P, shapes: list[str]):
    """
    Asserts that the center points of the shapes, when projected to the direction of the normal vector, coincide at the same location.
    Runs test for ANY combination of shape instances with the given name `shapes`.
    """
def assert_components(shapes: list[str]):
    """
    Asserts that the scene contains exactly the specified shape instances.
    """
