import sys
from pathlib import Path
from engine.constants import ENGINE_MODE, PROJ_DIR
assert ENGINE_MODE in ["exposed_v2", "exposed"], ENGINE_MODE
prompts_root = Path(PROJ_DIR) / 'scripts' / 'prompts'
sys.path.append(prompts_root.as_posix())

from scripts.prompts.helper import *
import drjit as dr
import mitsuba as mi
from scripts.prompts.impl_utils import run
from scripts.prompts.mi_helper import execute_from_preset
if ENGINE_MODE == "exposed_v2":
    import scripts.prompts.mesh_helper  # FIXME
from scripts.prompts.impl_preset import core
import math
import trimesh
import numpy as np

save_dir = Path(PROJ_DIR) / 'logs' / Path(__file__).stem
save_dir.mkdir(parents=True, exist_ok=True)
assets_dir = save_dir / 'assets'
assets_dir.mkdir(parents=True, exist_ok=True)


@register()
def scene() -> Shape:
    s1 = primitive_call('cube', color=(0.8, 0.8, 0.8), shape_kwargs={'scale': (2, 2, 2)})
    s2 = primitive_call('sphere', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 1})
    s3 = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 1, 'p0': (-0.5, -1, 0), 'p1': (0.5, 1, 0)})
    s1 = transform_shape(s1, translation_matrix((0, 0, 0)))
    s2 = transform_shape(s2, scale_matrix((1, 2, 1), origin=(0, 0, 0)))
    s2 = transform_shape(s2, translation_matrix((2, 0, 0)))
    s3 = transform_shape(s3, translation_matrix((4, 0, 0)))
    # return concat_shapes(s1, s2, s3)

    s4 = primitive_call('cone', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': 1, 'p0': (-0.5, -1, 0), 'p1': (0.5, 1, 0)})
    s4 = transform_shape(s4, translation_matrix((6, 0, 0)))

    return concat_shapes(s1, s2, s3, s4)


if __name__ == '__main__':
    shape = scene()
    print(shape)
    core(engine_modes=['mesh'], overwrite=True, save_dir=save_dir.as_posix())