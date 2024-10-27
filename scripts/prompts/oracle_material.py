from helper import *


"""
a simple sphere
"""


@register()
def simple_sphere() -> Shape:
    return primitive_call('sphere', shape_kwargs={'radius': 0.5}, bsdf_kwargs={'base_color': (.4, .2, .1), 'metallic': .7})


if __name__ == "__main__":
    from pathlib import Path
    from impl_utils import run
    from tu.loggers.utils import print_vcv_url
    root = Path(__file__).parent.parent
    save_dir = root / 'outputs' / Path(__file__).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    print_vcv_url(save_dir.as_posix())
    save_dir = Path(__file__).parent.parent / 'outputs' / Path(__file__).stem / 'renderings'
    save_dir.mkdir(exist_ok=True, parents=True)

    print('attempt 1', library_call('simple_sphere'))
    print('attempt 2', library_call('simple_sphere'))

    # _ = run('simple_sphere', save_dir=save_dir.as_posix(), preset_id='indoors_no_window', num_views=2)
    from impl_preset import core
    core(engine_modes=[], overwrite=True,
         save_dir=save_dir.as_posix())

    exit()

# https://github.com/mitsuba-renderer/mitsuba3/blob/7acc78514ca9e7b503d7011f189ad8fa62682e4a/include/mitsuba/render/ior.h#L16
# https://github.com/mitsuba-renderer/mitsuba3/blob/7acc78514ca9e7b503d7011f189ad8fa62682e4a/src/bsdfs/conductor.cpp#L101


"""
a desk with some papers on it
"""


@register("a simple sphere to demonstrate materials")
def simple_sphere() -> Shape:
    shape = primitive_call('sphere', color=(.4, .2, .1), scale=1)
    shape[0]['bsdf'] = {
        'type': 'dielectric',
        'int_ior': 'water',
        'ext_ior': 'air',
    }
    shape[0]['bsdf'] = {
        'type': 'thindielectric',
        'int_ior': 'bk7',
        'ext_ior': 'air'
    }
    shape[0]['bsdf'] = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': 0.1,
        'int_ior': 'bk7',
        'ext_ior': 'air'
    }
    shape[0]['bsdf'] = {
        'type': 'conductor',
        'material': 'Au'
    }
    shape[0]['bsdf'] = {
        'type': 'roughconductor',
        'material': 'Al',
        'distribution': 'ggx',
        'alpha_u': 0.05,
        'alpha_v': 0.3
    }
    shape[0]['bsdf'] = {
        'type': 'hair',
        'eumelanin': 0.2,
        'pheomelanin': 0.4
    }
    shape[0]['bsdf'] = {
        'type': 'plastic',
        'diffuse_reflectance': {
            'type': 'rgb',
            'value': [0.1, 0.27, 0.36]
        },
        'int_ior': 1.9
    }
    shape[0]['bsdf'] = {
        'type': 'roughplastic',
        'distribution': 'beckmann',
        'int_ior': 1.61,
        'diffuse_reflectance': {
            'type': 'rgb',
            'value': 0
        }
    }
    shape[0]['bsdf'] = {
        'type': 'pplastic',
        'diffuse_reflectance': {
            'type': 'rgb',
            'value': [0.05, 0.03, 0.1]
        },
        'alpha': 0.06
    }
    shape[0]['bsdf'] = {
        'type': 'principled',
        'base_color': {
            'type': 'rgb',
            'value': [1.0, 1.0, 1.0]
        },
        # 'metallic': 0.7,
        # 'specular': 0.6,
        # 'roughness': 0.2,
        # 'spec_tint': 0.4,
        # 'anisotropic': 0.5,
        # 'sheen': 0.3,
        # 'sheen_tint': 0.2,
        # 'clearcoat': 0.6,
        # 'clearcoat_gloss': 0.3,
        # 'spec_trans': 0.4
    }
    # shape[0]['bsdf'] = {
    #     'type': 'principledthin',
    #     'base_color': {
    #         'type': 'rgb',
    #         'value': [0.7, 0.1, 0.1]
    #     },
    #     'roughness': 0.15,
    #     'spec_tint': 0.1,
    #     'anisotropic': 0.5,
    #     'spec_trans': 0.8,
    #     'diff_trans': 0.3,
    #     'eta': 1.33
    # }
    return shape


