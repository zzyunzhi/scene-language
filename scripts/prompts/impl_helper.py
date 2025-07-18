from engine.utils.graph_utils import calculate_node_depths
import json
import inspect
from typing import Literal
from _shape_utils import compute_bbox, primitive_call
from dsl_utils import set_seed
from type_utils import T
from shape_utils import concat_shapes
from mi_helper import box_fn, shap_e_fn, primitive_box_fn
import hashlib
import engine_utils
import trimesh
import yaml
from pathlib import Path
import numpy as np
from math_utils import identity_matrix
from engine.constants import PROJ_DIR, ENGINE_MODE
from engine.utils.argparse_utils import setup_save_dir


def generate_prompt_key(base_string: str = 'prompt_kwargs'):
    hash_object = hashlib.md5(base_string.encode())
    return f"{base_string}_{hash_object.hexdigest()[:8]}"


PROMPT_KEY = generate_prompt_key()  # prompt_kwargs_29fc3136


def generate_mesh_key(name: str, kwargs: dict):
    hash_object = hashlib.md5(json.dumps({k: kwargs[k] for k in sorted(kwargs.keys())}).encode())
    return f'{name}_{hash_object.hexdigest()}'

orig_primitive_call = engine_utils.inner_primitive_call
# should NOT be changed by `make_new_library`


def make_new_library(library, library_equiv, tree_depth: int, root: str, engine_mode: Literal['lmd', 'neural', 'omost', 'loosecontrol', 'box', 'densediffusion', 'mesh', 'all']):
    decode_docstring = next(iter(library.values()))['docstring'].startswith('{')
    if decode_docstring and engine_mode not in ['interior', 'exterior', 'box', 'mesh', 'all']:
        assert tree_depth == -1, tree_depth
    if engine_mode == 'mesh':
        import scripts.prompts.mesh_helper  # overwrites primitive_call

    if engine_mode == 'mesh':
        library_dir = setup_save_dir(Path(PROJ_DIR) / 'logs' / 'library', log_unique=True)
        (library_dir / 'assets').mkdir(parents=True, exist_ok=True)
        scene_list: list[tuple[str, dict[str, str | None | T]]] = []

        active_eid = None

    def make_parent_target(_name):
        # print(f'[INFO] parent target: {_name=}')

        def target(**kwargs):
            cur_library = library.copy()
            library.clear()
            library.update(new_library)
            # with set_seed(0):
            shape = orig_library[_name]['__target__'](**kwargs)
            library.clear()
            library.update(cur_library)
            return shape

        return {'__target__': target, 'docstring': library[_name]['docstring'], 'hist_calls': [], 'last_call': None}

    def make_target(_name):
        # print(f'[INFO] target: {_name=}')
        # TODO maybe include non-numeric arguments (e.g., color) into prompt
        docstring = orig_library[_name]['docstring']
        prompt = docstring if not decode_docstring else json.loads(docstring)['prompt']

        def target(*args, **kwargs):
            cur_library = library.copy()
            library.clear()
            library.update(orig_library)
            # with set_seed(0):
            engine_utils.inner_primitive_call = orig_primitive_call  # FIXME not tested for engine_mode != 'mesh'
            shape = orig_library[_name]['__target__'](*args, **kwargs)
            engine_utils.inner_primitive_call = primitive_call_target  # FIXME not tested for engine_mode != 'mesh'
            box = compute_bbox(shape)
            library.clear()
            library.update(cur_library)

            sig = inspect.signature(orig_library[_name]['__target__'])
            complete_kwargs = {**{n: arg for n, arg in zip(sig.parameters, args)}, **kwargs}

            if engine_mode == 'neural':
                return shap_e_fn(prompt=prompt, scale=box.sizes, center=box.center, enforce_centered_origin=False)
            if engine_mode in ['loosecontrol', 'box']:
                return box_fn(prompt=prompt, scale=box.sizes, center=box.center, enforce_centered_origin=False,
                              shape_type='cube', kwargs=complete_kwargs)
            if engine_mode in ['lmd', 'omost', 'densediffusion', 'migc']:
                if len(shape) > 1:
                    return box_fn(prompt=prompt, scale=box.sizes, center=box.center, enforce_centered_origin=False,
                                  shape_type='cube', kwargs=complete_kwargs)
                else:
                    return primitive_box_fn(prompt=prompt, shape=shape, kwargs=complete_kwargs)  # use this if possible as bounding box loses orientation information
            if engine_mode == 'all':
                # must use concat_shapes(*(... for ...)) not concat_shapes(... for ...)
                return concat_shapes(*(primitive_box_fn(prompt=prompt, shape=[s], kwargs=complete_kwargs) for s in shape))
            if engine_mode in ['gala3d', 'exterior', 'interior', 'mesh']:
                if decode_docstring:
                    docstring_decoded = json.loads(docstring)
                    is_exterior = docstring_decoded['is_exterior']
                    yaw = docstring_decoded.get('yaw', 0)  # legacy
                    negative_prompt = docstring_decoded.get('negative_prompt', '')
                else:
                    is_exterior = False
                    yaw = 0
                    negative_prompt = ''
                if engine_mode == 'exterior':
                    return shape
                elif engine_mode == 'interior':
                    return [] if is_exterior else shape
                elif engine_mode == "mesh":
                    nonlocal active_eid
                    active_eid = active_eid + 1 if active_eid is not None else 0
                    asset_path = library_dir / 'assets' / f"{generate_mesh_key(_name, complete_kwargs)}.ply"

                    # object coordinate of current node = world coordinate in saved mesh
                    # requires applying leaf transformation saved in `to_world` for all descendants
                    if not asset_path.exists():
                        # mesh = trimesh.util.concatenate([trimesh.load(s['filename']).apply_transform(s['to_world']) for s in shape])

                        mesh = []
                        for s in shape:
                            m = trimesh.load(s['filename']).apply_transform(s['to_world'])
                            colors = np.ones((len(m.vertices), 4))
                            colors[:, :3] = s["bsdf"]["reflectance"]["value"]
                            m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=colors)
                            mesh.append(m)
                        mesh = trimesh.util.concatenate(mesh)
                        mesh.export(asset_path)
                    scene_list.append((_name, {
                        "description": prompt, 
                        "mesh": asset_path.as_posix(),
                        "eid": active_eid,
                    }))
                    return primitive_box_fn(prompt=prompt, shape=[{"type": "ply", "filename": asset_path.as_posix(), "to_world": identity_matrix()}], kwargs=complete_kwargs, eid=active_eid)
                extra_info = {'is_exterior': is_exterior, 'yaw': yaw, 'negative_prompt': negative_prompt}
                if len(shape) > 1:
                    return box_fn(prompt=prompt, scale=box.sizes, center=box.center, enforce_centered_origin=False,
                                  shape_type='cube', kwargs=complete_kwargs, **extra_info)
                else:
                    return primitive_box_fn(prompt=prompt, shape=shape, kwargs=complete_kwargs, **extra_info)
            raise NotImplementedError(engine_mode)

        return {'__target__': target, 'docstring': docstring, 'hist_calls': [], 'last_call': None}

    if decode_docstring:
        is_leaf = lambda _name: json.loads(orig_library[_name]['docstring'])['is_leaf']
        is_parent = lambda _name: json.loads(orig_library[_name]['docstring'])['is_parent']
    else:
        is_leaf = lambda _name: (node_depths[_name] <= tree_depth and len(node.children) == 0) or node_depths[_name] == tree_depth
        is_parent = lambda _name: not is_leaf(_name) and node_depths[_name] < tree_depth

    node_depths = calculate_node_depths(library_equiv, root)
    print(f'{node_depths=}')
    orig_library = library.copy()
    new_library = {}
    for name, node in library_equiv.items():
        if is_leaf(name):
            new_library[name] = make_target(name)
        elif is_parent(name):
            new_library[name] = make_parent_target(name)  # library[name]
        else:
            pass

    def primitive_call_target(_name, *args, **kwargs):
        sig = inspect.signature(orig_primitive_call)
        complete_kwargs = {**{n: arg for n, arg in zip(sig.parameters, args)}, **kwargs}
        if PROMPT_KEY not in complete_kwargs:
            prompt = _name
            is_exterior = False
            yaw = 0
            negative_prompt = ''
        else:
            prompt = complete_kwargs[PROMPT_KEY]['prompt']
            negative_prompt = complete_kwargs[PROMPT_KEY].get('negative_prompt', '')
            is_exterior = complete_kwargs[PROMPT_KEY]['is_exterior']
            yaw = complete_kwargs[PROMPT_KEY].get('yaw', 0)  # legacy
            _ = complete_kwargs.pop(PROMPT_KEY)
        shape = orig_primitive_call(_name, **complete_kwargs)
        assert len(shape) == 1, shape
        if engine_mode == 'exterior':
            return shape
        if engine_mode == 'interior':
            return [] if is_exterior else shape
        if engine_mode == 'mesh':
            nonlocal active_eid
            active_eid = active_eid + 1 if active_eid is not None else 0
            scene_list.append((
                _name,
                {
                    "description": prompt,
                    "mesh": shape[0]['filename'],
                    "eid": active_eid,
                }
            ))
            return primitive_box_fn(prompt=prompt, shape=shape, kwargs=complete_kwargs, eid=active_eid)
        extra_info = {} if engine_mode != 'gala3d' else {'is_exterior': is_exterior, 'yaw': yaw, 'negative_prompt': negative_prompt}
        return primitive_box_fn(prompt=prompt, shape=shape, kwargs=complete_kwargs, **extra_info)
    engine_utils.inner_primitive_call = primitive_call_target

    # def primitive_call_target(_name: Literal['sphere', 'cube'], **kwargs):
    #     shape = orig_primitive_call_target(_name, **kwargs)
    #     assert len(shape) == 1, shape
    #     return primitive_box_fn(prompt=kwargs.get(PROMPT_KEY, _name), shape=shape, kwargs=kwargs)
    #
    # orig_primitive_call_target = primitive_call.fn
    # primitive_call.fn = primitive_call_target

    if engine_mode == 'mesh':
        scene = new_library[root]['__target__']()
        if len(scene) != len(scene_list):
            print(f'[WARNING] len(scene) != len(scene_list), {len(scene)=}, {len(scene_list)=}')
        eid_to_scene_list_entry = {entry[1]["eid"]: entry for entry in scene_list}
        if len(eid_to_scene_list_entry) != len(scene_list):
            import ipdb; ipdb.set_trace()

        updated_scene_list = []
        for entry in scene:
            eid = entry["info"]["eid"]
            scene_list_entry = eid_to_scene_list_entry[eid]
            updated_scene_list.append((f"{scene_list_entry[0]}_{eid:03d}", {**scene_list_entry[1], "to_world": entry["to_world"].tolist()}))

        # old code snippet that disallows 1:many entry correspondence between `scene_list` and `scene`
        # which is violated e.g.:
        # ```python
            # def element_loop_fn(j) -> Shape:
            #     # ...
            #     return transform_shape(element, translation_matrix((x, y_pos, z)))  # Same element, different positions
            # return loop(4, element_loop_fn)
        # ```

        # eid_to_scene_entry = {entry["info"]["eid"]: entry for entry in scene}
        # if len(eid_to_scene_entry) != len(scene):
        #     import ipdb; ipdb.set_trace()
        # updated_scene_list = []
        # for eid in eid_to_scene_entry:
        #     scene_list_entry = eid_to_scene_list_entry[eid]
        #     scene_list_entry = (f"{scene_list_entry[0]}_{eid:03d}", scene_list_entry[1])
        #     scene_list_entry[1]['to_world'] = eid_to_scene_entry[eid]['to_world'].tolist()
        #     updated_scene_list.append(scene_list_entry)
        with open(library_dir / 'layout.yaml', 'w') as f:
            yaml.dump(dict(updated_scene_list), f)
        print(f'[INFO] Saved scene as dictionary to {library_dir / "layout.yaml"}')

    if engine_mode == "mesh" and ENGINE_MODE != 'exposed_v2':
        from scripts.prompts.mi_helper import impl_primitive_call
        from scripts.prompts._shape_utils import primitive_call
        primitive_call.implement(impl_primitive_call)

    return new_library
