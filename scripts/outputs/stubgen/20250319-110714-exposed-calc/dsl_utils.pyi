from type_utils import Shape

__all__ = ['register', 'library_call']

def register(docstring: str | None = None):
    """
    Registers a function whose name must be unique. You can pass in a docstring (optional).

    Every function you register MUST be invoked via `library_call`, and cannot be invoked directly via the function name.
    """
def library_call(func_name: str, **kwargs) -> Shape:
    """
    Call a function from the library and return its outputs. You are responsible for registering the function with `register`.

    Args:
        func_name (str): Function name.
        **kwargs: Keyword arguments passed to the function.
    """
