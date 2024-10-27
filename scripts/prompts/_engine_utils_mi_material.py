from typing import Literal, Union
from type_utils import Shape, P
import numpy as np
from _engine_utils_exposed import primitive_call as _primitive_call
from engine.constants import ENGINE_MODE
assert ENGINE_MODE == 'mi_material', ENGINE_MODE


__all__ = ["primitive_call"]

# https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/bsdfs/principled.cpp


def primitive_call(
        name: Literal["cube", "sphere", "cylinder"],
        shape_kwargs: dict[str, Union[float, P]],
        bsdf_kwargs: dict[str, Union[float, P]],
) -> Shape:
    """
    Constructs a primitive shape.

    Args:
        name: str - 'cube', 'sphere', or 'cylinder'.
        shape_kwargs: dict[str, Any] - keyword arguments for the primitive shape.
            - For 'cube': {'scale': P} - 3-tuple of floats for scaling along x, y, z axes.
            - For 'sphere': {'radius': float} - radius of the sphere.
            - For 'cylinder': {'radius': float, 'p0': P, 'p1': P}
                - radius: float - radius of the cylinder.
                - p0: P - 3-tuple of floats for the start point of the cylinder's centerline.
                - p1: P - 3-tuple of floats for the end point of the cylinder's centerline.
        bsdf_kwargs: dict[str, Any] - keyword arguments for the principled BSDF.
            - base_color: P - RGB color in range [0, 1]^3. (Default:(1, 1, 1))
            - roughness: float - Controls the roughness parameter of the main specular lobes. (Default:0.5)
            - anisotropic: float - Controls the degree of anisotropy. (0.0 : isotropic material) (Default:0.0)
            - metallic: float - The "metallicness" of the model. (Default:0.0)
            - spec_trans: float - Blends BRDF and BSDF major lobe. (1.0: only BSDF
                response, 0.0 : only BRDF response.) (Default: 0.0)
            - specular: float - Controls the Fresnel reflection coefficient. (Default:0.5)
            - spec_tint: float - The fraction of `base_color` tint applied onto the dielectric reflection
                lobe. (Default:0.0)
            - sheen: float - The rate of the sheen lobe. (Default:0.0)
            - sheen_tint: float - The fraction of `base_color` tint applied onto the sheen lobe. (Default:0.0)
            - flatness: float - Blends between the diffuse response and fake subsurface approximation based
                on Hanrahan-Krueger approximation. (0.0:only diffuse response, 1.0:only
                fake subsurface scattering.) (Default:0.0)
            - clearcoat: float - The rate of the secondary isotropic specular lobe. (Default:0.0)
            - clearcoat_gloss: float - Controls the roughness of the secondary specular lobe. Clearcoat response
                gets glossier as the parameter increases. (Default:0.0)
            - diffuse_reflectance_sampling_rate: float - The rate of the cosine hemisphere reflection in sampling. (Default:1.0)
            - main_specular_sampling_rate: float - The rate of the main specular lobe in sampling. (Default:1.0)
            - clearcoat_sampling_rate: float - The rate of the secondary specular reflection in sampling. (Default:1.0)

    Returns:
        Shape - the primitive shape.

    Examples:
        - `primitive_call('cube', shape_kwargs={'scale': (1, 2, 1)}, bsdf_kwargs={})`
          Returns a cube with corners (-0.5, -1, -0.5) and (0.5, 1, 0.5).
        - `primitive_call('sphere', shape_kwargs={'radius': 0.5}, bsdf_kwargs={})`
          Returns a sphere with radius 0.5, with bounding box corners (-0.5, -0.5, -0.5) and (0.5, 0.5, 0.5).
        - `primitive_call('cylinder', shape_kwargs={'radius': 0.5, 'p0': (0, 0, 0), 'p1': (0, 1, 0)}, bsdf_kwargs={})`
          Returns a cylinder with bounding box corners (-0.5, 0, -0.5) and (0.5, 1, 0.5).
    """
    shape = _primitive_call(name, shape_kwargs)  # centered at origin
    for elem in shape:
        elem['bsdf'] = {
            'type': 'principled',
            'base_color': {'type': 'rgb', 'value': np.asarray(bsdf_kwargs.get('base_color', (1, 1, 1))).tolist()},
            **{k: v for k, v in bsdf_kwargs.items() if k != 'base_color'},
        }  # not exposing eta
    return shape
