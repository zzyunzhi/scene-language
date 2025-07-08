from type_utils import Shape
from engine.constants import ENGINE_MODE


__all__ = ["primitive_call"]


if ENGINE_MODE == 'mi':
    from _engine_utils_mi import primitive_call as _primitive_call
elif ENGINE_MODE == 'neural':
    from _engine_utils_neural import primitive_call as _primitive_call
elif ENGINE_MODE == 'lmd':
    from _engine_utils_lmd import primitive_call as _primitive_call
elif ENGINE_MODE == 'minecraft':
    from _engine_utils_minecraft import primitive_call as _primitive_call
elif ENGINE_MODE == 'mi_material':
    from _engine_utils_mi_material import primitive_call as _primitive_call
elif ENGINE_MODE == 'mi_from_minecraft':
    from _engine_utils_mi_from_minecraft import primitive_call as _primitive_call
elif ENGINE_MODE == 'exposed':
    from _engine_utils_exposed import primitive_call as _primitive_call
elif ENGINE_MODE == 'exposed_v2':
    from _engine_utils_exposed_v2 import primitive_call as _primitive_call
else:
    raise NotImplementedError(ENGINE_MODE)


def primitive_call(name, *args, **kwargs) -> Shape:
    # inner_primitive_call may be updated in `impl_helper.make_new_library`
    return inner_primitive_call(name, *args, **kwargs)


def inner_primitive_call(name, *args, **kwargs) -> Shape:
    kwargs = {k: v for k, v in kwargs.items() if k != 'prompt_kwargs_29fc3136'}
    return _primitive_call(name, *args, **kwargs)
