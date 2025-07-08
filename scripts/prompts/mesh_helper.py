import trimesh
from type_utils import T, Shape, Box, P
from math_utils import _scale_matrix, translation_matrix, rotation_matrix, identity_matrix, align_vectors
from typing import Literal, Callable, Union, Optional, List
import numpy as np
from engine.constants import PROJ_DIR
from pathlib import Path
from _shape_utils import placeholder, primitive_call, transform_shape, compute_bbox
import mitsuba as mi

assets_dir = Path(PROJ_DIR) / 'assets'
assets_dir.mkdir(parents=True, exist_ok=True)

ASSETS_PATHS = {
    'cube': assets_dir / 'cube.ply',
    'sphere': assets_dir / 'sphere.ply',
    'cylinder': assets_dir / 'cylinder.ply',
    'cone': assets_dir / 'cone.ply',
}

def create_assets():
    if all(p.exists() for p in ASSETS_PATHS.values()):
        print(f'[INFO] Assets already exist under {assets_dir}, skipping')
        return
    box = trimesh.creation.box(extents=(1, 1, 1))
    box.export(ASSETS_PATHS['cube'].as_posix())
    print(box.bounds)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1)
    sphere.export(ASSETS_PATHS['sphere'].as_posix())
    print(sphere.bounds)
    cylinder = trimesh.creation.cylinder(radius=1, height=1, 
                                         transform=translation_matrix((0, 0.5, 0)) @ align_vectors(np.array([0, 1, 0]), np.array([0, 0, 1])))
    cylinder.export(ASSETS_PATHS['cylinder'].as_posix())
    print(cylinder.bounds)
    cone = trimesh.creation.cone(radius=1, height=1, 
                                 transform=align_vectors(np.array([0, 1, 0]), np.array([0, 0, 1])))
    cone.export(ASSETS_PATHS['cone'].as_posix())
    print(cone.bounds)
create_assets()

cube_fn: Callable[[Union[float, P], P], Shape] = lambda scale, color=(1, 1, 1): [{
    'type': 'ply', 'filename': ASSETS_PATHS['cube'].as_posix(),
    'to_world': _scale_matrix(scale, enforce_uniform=False),
    'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': np.asarray(color[:3]).clip(0, 1)}},
    'info': {'stack': []}
}]

cylinder_fn: Callable[[float, P, P], Shape] = lambda radius, p0, p1, color=(1, 1, 1): [{
    'type': 'ply', 'filename': ASSETS_PATHS['cylinder'].as_posix(),
    'to_world': translation_matrix(np.array(p0)) @ align_vectors(np.array(p1) - np.array(p0), np.array([0, 1, 0])) @ _scale_matrix([radius, np.linalg.norm(np.array(p1) - np.array(p0)), radius], enforce_uniform=False),
    'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': np.asarray(color[:3]).clip(0, 1)}},
    'info': {'stack': []}
}]

sphere_fn: Callable[[P, Union[float, P]], Shape] = lambda radius, color=(1, 1, 1): [{
    'type': 'ply', 'filename': ASSETS_PATHS['sphere'].as_posix(),
    'to_world': _scale_matrix(radius),
    'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': np.asarray(color[:3]).clip(0, 1)}},
    'info': {'stack': []}
}]

cone_fn: Callable[[float, P, P], Shape] = lambda radius, p0, p1, color=(1, 1, 1): [{
    'type': 'ply', 'filename': ASSETS_PATHS['cone'].as_posix(),
    'to_world': translation_matrix(np.array(p0)) @ align_vectors(np.array(p1) - np.array(p0), np.array([0, 1, 0])) @ _scale_matrix([radius, np.linalg.norm(np.array(p1) - np.array(p0)), radius], enforce_uniform=False),
    'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': np.asarray(color[:3]).clip(0, 1)}},
    'info': {'stack': []}
}]

def impl_primitive_call():
    def fn(name: Literal['sphere', 'cube', 'cylinder', 'cone'], **kwargs):
        return {
            'cube': cube_fn, 'sphere': sphere_fn, 'cylinder': cylinder_fn, 'cone': cone_fn,
        }[name](**kwargs)

    return fn


primitive_call.implement(impl_primitive_call)