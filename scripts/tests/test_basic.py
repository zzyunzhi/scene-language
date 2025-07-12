import sys
from pathlib import Path
from engine.constants import PROJ_DIR  # this will also initialize mitsuba for rendering
sys.path.append((Path(PROJ_DIR) / "scripts/prompts").as_posix())
from helper import *


# Tip: If you want to test a different program, simply replace the following program block with your own.

###### Program starts here ######
@register()
def cube():
    return primitive_call('cube', color=(0.2, 0.2, 0.2), shape_kwargs={'scale': (5, 0.1, 5)})

@register()
def sphere():
    return primitive_call('sphere', color=(0.8, 0.2, 0.2), shape_kwargs={'radius': 0.5})

@register()
def scene():
    cube = library_call('cube')
    sphere = library_call('sphere')
    return concat_shapes(cube, transform_shape(sphere, translation_matrix((0, 1, 0))))

###### Program ends here ######

if __name__ == "__main__":
    from impl_preset import core
    save_dir = Path(__file__).parent / Path(__file__).stem
    if save_dir.exists():
        print(f'[INFO] {save_dir} will be overwritten!')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(save_dir.as_posix())
    core(engine_modes=[], overwrite=True, save_dir=save_dir.as_posix())
