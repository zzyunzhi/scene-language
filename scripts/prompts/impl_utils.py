from pathlib import Path

import json
# import mitsuba.mitsuba_scalar_rgb_ext
from scipy.spatial.transform import Rotation as R

from mi_helper import execute_from_preset, project, render_depth, compute_best_view_from_z_from_top
import mitsuba as mi
try:
    from engine.utils.omost_utils import MyCanvas
    from omost_helper import run_pipe as run_omost
    from controlnet_helper import clear_controlnet_model, run_pipe as run_controlnet
    from loosecontrol_helper import run_pipe as run_loosecontrol
    from dense_diffusion_helper import run_pipe as run_densediffusion
    from scripts.exp.icl_0429.prompts.lmd_plus_helper import run_pipe as run_lmd
    from migc_helper import run_pipe as run_migc
    from gala3d_helper import run_pipe as run_gala3d, run_pipe_post as run_gala3d_post, run_pipe_post_animation as run_gala3d_post_animation
except:
    print("[WARNING] Failed to import neural pipelines.")
    # import traceback; traceback.print_exc()
from dsl_utils import library, set_fake_call_enabled, set_seed, set_track_history_enabled, clear_history, animation_library_call
from math_utils import _scale_matrix, translation_matrix, rotation_matrix, identity_matrix, align_vectors
from _shape_utils import transform_shape, compute_bbox, Hole  # don't use the library here
import inspect
import traceback
from engine.constants import ENGINE_MODE, DEBUG
from engine.utils.docstring_utils import describe_color
from helper import library_call, T
from prompt_helper import load_program
from typing import Union, Optional
import numpy as np
import ast
import sys
from contextlib import contextmanager
from PIL import Image
import imageio
import os


def run(scene_name: str, preset_id: str, save_dir: Union[str, None] = None,
        engine_mode: str = ENGINE_MODE,
        # normalization: Union[None, T] = None,
        # sensors: Union[None, dict[str, mi.Sensor]] = None,
        prev_out: Optional[dict] = None,
        num_views: int = 6,
        overwrite: bool = DEBUG,
        execute_controlnet: bool = False,
        save_suffix: Optional[str] = None,
        new_library: Optional[dict[str, dict]] = None) -> dict:
    if new_library is None:
        new_library = library
    if scene_name not in new_library:
        print(f'[ERROR] {scene_name=} not in library')
        return None
    prefix = f'{engine_mode}_{scene_name}_{preset_id}'
    if save_suffix is not None:
        prefix = f'{prefix}_{save_suffix}'
    # print(f'running with {prefix=}')

    if save_dir is None:
        save_dir = Path(__file__).parent / 'renderings'
    else:
        save_dir = Path(save_dir)

    def save_gif(name, paths: list[Path], fps=4.):
        if name in seq_name_to_frame_paths:
            print(f'[INFO] {name=} already will be overwritten')
        seq_name_to_frame_paths[f'{prefix}_{name}'] = paths
        if len(paths) == 0:
            return
        # if len(paths) != len(out['sensors']):  # TODO
        #     print(f'[ERROR] {name=} {len(paths)=} {len(out["sensors"])}')
        imageio.mimsave((save_dir / f'{prefix}_{name}.gif').as_posix(), [np.asarray(Image.open(p.as_posix())) for p in paths], fps=fps, loop=0)

    final_seq_name = 'rendering_traj'
    final_seq_path: Path = save_dir / f'{prefix}_{final_seq_name}.gif'

    seq_name_to_frame_paths: dict[str, list[Path]] = {}
    save_frame_dirs = []
    out = {} if prev_out is None else {**prev_out}
    for frame_ind, t in enumerate(np.linspace(0, 1, num=1, endpoint=False)):
        if t > 0:
            break
        assert frame_ind == 0, f'{frame_ind=} {t=} not supported'
        # try:
        #     frame = library_call(scene_name, t=t)
        # except Exception:
        #     if t > 0:  # only need to run the first frame, generate all traj images once
        #         break
        # frame = library_call(scene_name)
        with set_seed(0):
            frame = new_library[scene_name]['__target__']()
        save_frame_dir = save_dir / f'{prefix}_frame_{frame_ind:02d}'
        save_frame_dir.mkdir(parents=True, exist_ok=True)
        save_frame_dirs.append(save_frame_dir)
        if overwrite or not final_seq_path.exists():
            out = execute_from_preset(frame, save_dir=save_frame_dir.as_posix(), preset_id=preset_id,
                                      # normalization=out['normalization'],
                                      # sensors=out['sensors'],
                                      timestep=(0, num_views),
                                      prev_out=prev_out,
                                      )
            # traj_types = ['forward_facing', '360_view']
            # fov_types = ['changing', 'fixing']
            # traj_paths = []     # to use the unlink
            # for traj_type in traj_types:
            #     for fov_type in fov_types:
            #         individual_traj_paths = list(sorted(save_frame_dirs[0].glob(f'rendering_traj_{traj_type}_{fov_type}_[0-9][0-9][0-9].png')))
            #         imageio.mimsave((save_dir / f'{prefix}_rendering_traj_{traj_type}_{fov_type}.gif').as_posix(),  # mitsuba renderings of bounding boxes
            #                         [np.asarray(Image.open(p.as_posix())) for p in individual_traj_paths], fps=4, loop=0)
            #         traj_paths += individual_traj_paths

            # if engine_mode == 'lmd':
            #     tmp_sensors = {}
            #     for sensor_name, sensor in out['sensors'].items():
            #         if 'traj' not in sensor_name:
            #             tmp_sensors.update({sensor_name: sensor})
            #     out['sensors'] = tmp_sensors

            # commented out above so that we still render all frames in trajectory + additional sensors

            if engine_mode in ['lmd', 'omost', 'densediffusion', 'densesds', 'migc']:
                save_frame_boxes_dir = save_frame_dir / 'boxes'
                save_frame_boxes_dir.mkdir(parents=True, exist_ok=True)

                docstrings = []
                for s in frame:
                    if 'info' not in s or 'docstring' not in s['info']:
                        # print('[ERROR] docstring not found', s)
                        # func_name, _ = s['info']['stack'][0]
                        # docstring = library[func_name]['docstring']
                        docstring = None
                        # FIXME hack for a program like the following:
                        """
```python
@register('Torch of the Statue of Liberty')
def torch(scale: float = 1.0) -> Shape:
    torch_base = primitive_call('sphere', scale=scale*0.15, color=(1, 0.8, 0))
    flame_shape = transform_shape(library_call('flame', scale=scale), translation_matrix((0, scale*0.2, 0)))
    return concat_shapes(torch_base, flame_shape)
```
                        """
                        # torch_base won't have docstring assigned
                        # so we use the function name of its direct parent `"torch"`
                    else:
                        docstring = s['info']['docstring']
                        docstring = docstring.split(';')[0].lower()
                    docstrings.append(docstring)
                boxes, segm_maps, depth_maps = project(frame, save_dir=save_frame_boxes_dir.as_posix(),
                                                       normalization=out['normalization'], sensors=out['sensors'])
                if engine_mode == 'lmd':
                    for sensor_name, sensor_boxes in boxes.items():
                        sensor_boxes = [(s, b) for s, b in zip(docstrings, sensor_boxes) if np.all(b.sizes > 5) and s is not None]
                        if len(sensor_boxes) == 0:
                            import pdb; pdb.set_trace()
                        run_lmd(boxes=sensor_boxes,
                                prompt=library[scene_name]['docstring'].capitalize(),  # TODO
                                save_dir=save_frame_dir.as_posix(),
                                resolution=max(np.asarray(mi.traverse(out['sensors'][sensor_name])['film.size'])),
                                bg_prompt=None,
                                save_prefix=sensor_name,
                                overwrite=overwrite)
                        # hard-coded path and resolution for LMD
                        expected_path = Path(save_frame_dir) / f'{sensor_name}_00' / 'img_0.png'
                        # if not expected_path.exists():
                        #     Image.new('RGB', (1024, 1024), color='black').save(expected_path.as_posix())
                        # if Image.open(expected_path).size[0] == 512:  # FIXME legacy fix
                        #     print('[INFO] replacing with new black image of resolution 1024')
                        #     Image.new('RGB', (1024, 1024), color='black').save(expected_path.as_posix())
                elif engine_mode == 'migc':
                    for sensor_name, sensor_boxes in boxes.items():
                        sensor_boxes = [(s, b) for s, b in zip(docstrings, sensor_boxes) if np.all(b.sizes > 5) and s is not None]
                        run_migc(
                            boxes=sensor_boxes, prompt=library[scene_name]['docstring'],
                            save_dir=save_frame_dir.as_posix(),
                            resolution=max(np.asarray(mi.traverse(out['sensors'][sensor_name])['film.size'])),
                            save_prefix=sensor_name,
                            overwrite=overwrite,
                        )

                elif engine_mode == 'omost':
                    for sensor_name in out['sensors'].keys():
                        canvas = MyCanvas()
                        canvas.set_global_description(  # TODO ask LM to fill in
                            description=library[scene_name]['docstring'].capitalize() + '.',
                            detailed_descriptions=[],
                            tags='',
                            HTML_web_color_name='white'
                        )
                        for ind in range(len(frame)):
                            if docstrings[ind] is None:
                                continue
                            segm = segm_maps[sensor_name][ind]
                            depth = depth_maps[sensor_name][ind]
                            box = boxes[sensor_name][ind]
                            canvas.add_my_local_description(  # TODO ask LM to fill in
                                segm=segm, depth=depth, box=box,
                                description=docstrings[ind].capitalize() + '.',
                                detailed_descriptions=[],
                                tags='',
                                atmosphere='',
                                style='',
                                quality_meta='',
                                color=frame[ind]['bsdf']['reflectance']['value'],
                            )
                        run_omost(canvas=canvas, save_dir=save_frame_dir.as_posix(), save_prefix=sensor_name)
                elif engine_mode == 'densediffusion':
                    bg_prompt = 'in a field'
                    prompt = f'{library[scene_name]["docstring"]} {bg_prompt}, with {", ".join(docstrings)}'
                    for sensor_name in out['sensors'].keys():
                        run_densediffusion(prompt=prompt,  # TODO
                                           bg_prompt=bg_prompt,  # TODO
                                           docstrings=docstrings,
                                           segm_maps=segm_maps[sensor_name],
                                           depth_maps=depth_maps[sensor_name],
                                           save_dir=save_frame_dir.as_posix(), save_prefix=sensor_name,
                                           overwrite=overwrite,
                                           collapse_prompts=False,
                                           creg_=1,
                                           sreg_=1,
                                           )
                elif engine_mode == 'densesds':
                    raise NotImplementedError(engine_mode)
                else:
                    raise NotImplementedError(engine_mode)

            elif engine_mode == 'loosecontrol':
                save_frame_depth_dir = save_frame_dir / 'depth'
                save_frame_depth_dir.mkdir(parents=True, exist_ok=True)
                segm_maps, depth_maps = render_depth(frame, save_dir=save_frame_depth_dir.as_posix(),
                                                     normalization=out['normalization'], sensors=out['sensors'])
                for sensor_name in out['sensors'].keys():
                    run_loosecontrol(depth=depth_maps[sensor_name],
                                     segm=segm_maps[sensor_name],
                                     prompt=library[scene_name]['docstring'],
                                     save_dir=save_frame_dir.as_posix(), overwrite=overwrite,
                                     save_prefix=sensor_name)
            elif engine_mode == 'gala3d':
                box = compute_bbox(frame)
                box_radius = .5
                scale = box_radius * 2 / np.linalg.norm(box.sizes)
                min_box_radius = .1
                scale = max(scale, max([min_box_radius * 2 / np.linalg.norm(compute_bbox([s]).sizes) for s in frame]))
                box_radius = scale * np.linalg.norm(box.sizes) / 2
                # normalization = _scale_matrix(scale) @ translation_matrix((-box.center[0], box.sizes[1] / 2 - box.center[1], -box.center[2]))
                # changed after 20240908 16:43
                normalization = _scale_matrix(scale) @ translation_matrix((-box.center[0], -box.center[1], -box.center[2]))
                norm_frame = transform_shape(frame, normalization)
                exterior_flags = []
                yaws = []  # transform from MVDream (Objaverse) canonical space to program primitive canonical space
                docstrings = []
                negative_docstrings = []
                toworld_list = []
                for s in norm_frame:
                    exterior_flags.append(s['info']['is_exterior'])
                    yaws.append(s['info']['yaw'])
                    docstring = s['info']['docstring'].replace('_', ' ')
                    # if 'peasant' in docstring:  # FIXME hack...
                    #     if s['info']['kwargs']['height'] == 1.8:
                    #         docstring = docstring.replace('peasant', 'male peasant')
                    #     if s['info']['kwargs']['height'] == 1.6:
                    #         docstring = docstring.replace('peasant', 'female peasant')
                    # FIXME manually comment it out if not used
                    if 'color' in s['info']['kwargs']:
                        color = describe_color(s['info']['kwargs']['color'])
                        if color is not None:
                            docstring = docstring + ',' + color
                    docstrings.append(docstring)
                    negative_docstrings.append(s['info']['negative_prompt'])
                    toworld = s['to_world']
                    if s['type'] == 'cylinder':
                        toworld = toworld @ get_transform_cylinder(s)
                    toworld_list.append(toworld)
                scene_prompt = library[scene_name]['docstring']
                if scene_prompt.startswith('{'):
                    scene_prompt = json.loads(scene_prompt)['prompt']
                scene_prompt = scene_prompt.replace('_', ' ')

                # frame: program world frame
                # preset_frame: mitsuba sensor world frame
                # norm_frame: GALA3D world frame

                assert out['normalization'] is not None
                preset_frame = transform_shape(frame, out['normalization'])
                cano_ind = 0
                o2w = preset_frame[cano_ind]['to_world'].astype(np.float32)
                if preset_frame[cano_ind]['type'] == 'cylinder':
                    # must go through the same transformation
                    s = preset_frame[cano_ind]
                    o2w = o2w @ get_transform_cylinder(s)

                sensors = {}
                for sensor_name, sensor in out['sensors'].items():
                    c2w = np.asarray(mi.traverse(sensor)['to_world'].matrix).squeeze(0)
                    fov = np.asarray(mi.traverse(sensor)['x_fov']).astype(np.float32).item()  # assume square, in degree
                    sensors[sensor_name] = {'c2w': c2w, 'o2w': o2w, 'fov': fov}

                for i, sensor_name in enumerate(out['sensors'].keys()):
                    sensors[sensor_name].update({
                        'eye': out['sensor_info']['eyes'][i],
                        'target': out['sensor_info']['targets'][i],
                    })

                # from tu.utils.pose import assemble_rot_trans_np
                # from engine.third_party.gala3d.cam_utils import look_at
                # from engine.utils.camera_utils import orbit_camera
                # box = compute_bbox(norm_frame)
                #
                # num_frames = 6
                # elev = -20
                # radius = np.linalg.norm(box.sizes) / 2 * 2
                # target = box.center
                # azims = np.linspace(0, 360, num_frames, endpoint=False).tolist()
                # bestviews = [orbit_camera(elev, azim, radius=radius, target=target) for azim in azims]
                # bestview_fovs = [49.1] * len(bestviews)
                #
                # trans = toworld_list[cano_ind] @ np.linalg.inv(o2w)
                # print('wp2w transformation', trans)
                # # import ipdb; ipdb.set_trace()
                #
                # eye_ref = np.asarray(bestviews[0])
                # target_ref = np.asarray(box.center)
                # eye = out['sensor_info']['eyes'][0]
                # target = out['sensor_info']['targets'][0]
                # eye_trans = trans[:3, :3] @ eye + trans[:3, 3]
                # target_trans = trans[:3, :3] @ target + trans[:3, 3]
                # import ipdb; ipdb.set_trace()
                # print(np.round(eye_ref - eye_trans, 3))
                # print(np.round(target_ref - target_trans, 3))
                #
                # for i, sensor_name in enumerate(out['sensors'].keys()):
                #     eye = np.asarray(bestviews[i])
                #     target = np.asarray(box.center)
                # #     mat1 = np.asarray(mi.scalar_rgb.Transform4f.look_at(
                # #         origin=eye,
                # #         target=target,
                # #         up=[0, 1, 0]
                # #     ).matrix)
                # #     mat2 = assemble_rot_trans_np(look_at(campos=eye, target=target, opengl=True), eye).astype(np.float32)
                # #     print('mitsuba', mat1)
                # #     print('gala3d', mat2)
                #     sensors[sensor_name].update({
                #         # 'c2w': assemble_rot_trans_np(look_at(campos=eye, target=target, opengl=True), eye).astype(np.float32),
                #         'eye': eye,
                #         'target': target,
                #         # 'fov': fovs[i]
                #     })

                legacy_prefix = f'gala3d_{scene_name}_indoors_no_window'
                if save_suffix is not None:
                    legacy_prefix = f'{legacy_prefix}_{save_suffix}'
                legacy_save_dir = save_dir / f'{legacy_prefix}_frame_{frame_ind:02d}'

                run_gala3d(
                    docstrings=docstrings,
                    negative_docstrings=negative_docstrings,
                    exterior_flags=exterior_flags,
                    yaws=yaws,
                    toworld_list=toworld_list,
                    save_dir=legacy_save_dir.as_posix(),
                    overwrite=overwrite,
                    resolution=512,
                    cam_radius=box_radius * 4,
                    sensors=sensors,
                    prompt=scene_prompt,
                )

                run_gala3d_post(
                    load_dir=legacy_save_dir.as_posix(),
                    save_dir=save_frame_dir.as_posix(),
                    overwrite=overwrite,
                    resolution=1024,
                    sensors=sensors,
                )

                seq_frames = animation_library_call(new_library)

                if seq_frames is not None:
                    seq_box = compute_bbox(sum(seq_frames, []))
                    seq_normalization = _scale_matrix(scale) @ translation_matrix((-seq_box.center[0], -seq_box.center[1], -seq_box.center[2]))
                    norm_seq_frames = [transform_shape(seq_frame, seq_normalization) for seq_frame in seq_frames]
                    seq_box = compute_bbox(sum(norm_seq_frames, []))
                    toworld_list_list = []
                    for seq_frame in norm_seq_frames:
                        if len(seq_frame) != len(frame):
                            print('[ERROR] number of shapes mismatch', len(seq_frame), len(frame))
                            continue
                        toworld_list = []
                        for s in seq_frame:
                            toworld = s['to_world']
                            if s['type'] == 'cylinder':
                                toworld = toworld @ get_transform_cylinder(s)
                            toworld_list.append(toworld)
                        toworld_list_list.append(toworld_list)

                    cam_radius = np.linalg.norm(seq_box.sizes) * 2

                    run_gala3d_post_animation(
                        frames=toworld_list_list,
                        load_dir=legacy_save_dir.as_posix(),
                        save_dir=(save_frame_dir / 'animated').as_posix(),
                        cam_radius=cam_radius,
                        resolution=1024,
                    )
            elif engine_mode in ['mi', 'mi_from_minecraft', 'neural', 'mi_material', 'exposed', 'interior', 'exterior']:
                pass
            else:
                raise NotImplementedError(engine_mode)
    traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9].png')))
    if engine_mode == 'lmd':
        save_gif('primitives_rendering_traj', traj_paths)
        lmd_boxes_traj_paths = list(sorted(save_frame_dirs[0].glob('boxes/sensor_rendering_traj_[0-9][0-9][0-9]_shape_all.png')))
        save_gif('boxes_rendering_traj', lmd_boxes_traj_paths)
        lm_prompts_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_00/boxes.png')))  # TODO currently these images are not squared
        save_gif('prompts_rendering_traj', lm_prompts_traj_paths)
        lmd_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_00/img_0.png')))
        save_gif(final_seq_name, lmd_traj_paths)
    elif engine_mode == 'migc':
        save_gif('primitives_rendering_traj', traj_paths)
        migc_anno_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_output_anno.png')))
        save_gif('anno_rendering_traj', migc_anno_traj_paths)
        migc_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_output.png')))
        save_gif(final_seq_name, migc_traj_paths)
    elif engine_mode == 'gala3d':
        save_gif('primitives_rendering_traj', traj_paths)
        gala3d_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_out.png')))
        save_gif(final_seq_name, gala3d_traj_paths)
        gala3d_animated_traj_paths = list(sorted(save_frame_dirs[0].glob('animated/[0-9][0-9][0-9].png')))
        save_gif('animated_traj', gala3d_animated_traj_paths, fps=len(gala3d_animated_traj_paths) / 2)
    elif engine_mode == 'omost':
        save_gif('primitives_rendering_traj', traj_paths)
        omost_latents_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_latents.png')))
        save_gif('latents_rendering_traj', omost_latents_traj_paths)
        omost_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_image_00.png')))
        save_gif(final_seq_name, omost_traj_paths)
    elif engine_mode == 'loosecontrol':
        save_gif('primitives_rendering_traj', traj_paths)
        loosecontrol_depth_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_depth.png')))
        save_gif('depth_rendering_traj', loosecontrol_depth_traj_paths)
        loosecontrol_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_image.png')))
        save_gif(final_seq_name, loosecontrol_traj_paths)
    elif engine_mode == 'densediffusion':
        save_gif('primitives_rendering_traj', traj_paths)
        densediffusion_traj_paths = list(sorted(save_frame_dirs[0].glob('rendering_traj_[0-9][0-9][0-9]_feedforward.png')))
        save_gif(final_seq_name, densediffusion_traj_paths)
    else:
        save_gif(final_seq_name, traj_paths)

    # for first_frame_path in save_frame_dirs[0].glob('*.png'):
    #     images = [np.asarray(Image.open((save_frame_dir / first_frame_path.name).as_posix())) for save_frame_dir in save_frame_dirs]
    #     imageio.mimsave((save_dir / f'{prefix}_{first_frame_path.stem}.gif').as_posix(), images, fps=4, loop=0)

        # each frame is new t
        # for first_frame_path in save_frame_dirs[0].glob('*_00/img_0.png'):
        #     images = [np.asarray(Image.open((save_frame_dir / first_frame_path.relative_to(save_frame_dirs[0])).as_posix()))
        #               for save_frame_dir in save_frame_dirs]
        #     imageio.mimsave((save_dir / f'{prefix}_{first_frame_path.parent.name}.gif').as_posix(), images, fps=4, loop=0)

        # each frame is a new sensor

    if not DEBUG and execute_controlnet and engine_mode in ['mi', 'mi_material']:
        controlnet_input_dir = save_frame_dirs[0]
        if engine_mode in ['mi', 'mi_material']:
            # controlnet_input_images = list(controlnet_input_dir.glob('*traj*.png'))
            controlnet_input_images = traj_paths
        # elif engine_mode == 'lmd':
        #     # controlnet_input_images = list(controlnet_input_dir.glob('*traj*_00/img_0.png'))
        #     controlnet_input_images = lmd_traj_paths
        controlnet_output_dir = controlnet_input_dir / 'controlnet'
        controlnet_output_dir.mkdir(exist_ok=True)
        for controlnet in ['canny', 'softedge', 'seg', 'mlsd']:
            if overwrite or not (save_dir / f'{prefix}_controlnet_{controlnet}_rendering_traj.gif').exists():
                clear_controlnet_model()
                for frame_ind, image in enumerate(controlnet_input_images):
                    run_controlnet(image=image.as_posix(), prompt=scene_name, save_dir=controlnet_output_dir.as_posix(),
                                   save_prefix=f'rendering_traj_{frame_ind:03d}', overwrite=overwrite, model=controlnet)
            controlnet_traj_paths = list(sorted(controlnet_output_dir.glob(f'rendering_traj_[0-9][0-9][0-9]_{controlnet}_output.png')))
            save_gif(f'controlnet_{controlnet}_rendering_traj', controlnet_traj_paths)
    out['seq_name_to_frame_paths'] = seq_name_to_frame_paths
    out['final_frame_paths'] = seq_name_to_frame_paths[f'{prefix}_{final_seq_name}']
    return out


def align_vector_pair(a, b):
    # align b to a
    axis = np.cross(b, a)  # rotation axis
    if np.allclose(axis, 0):
        # choose an arbitrary perpendicular axis
        axis = np.array([1, 0, 0])  # or [0, 0, 1]

    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    ret = np.eye(4)
    ret[:3, :3] = R.from_rotvec(axis * angle).as_matrix()
    return ret


def get_transform_cylinder(s) -> T:
    height = np.linalg.norm(s['p1'] - s['p0'])
    ret = (
            translation_matrix(np.asarray(s['p0']).tolist())
            @ align_vector_pair((s['p1'] - s['p0']) / height, (0, 1, 0))
            @ translation_matrix((0, .5 * height, 0))
            @ _scale_matrix((s['radius'], .5 * height, s['radius']), enforce_uniform=False)
    )

    # m4 = translation_matrix(np.asarray(s['p0']).tolist())
    # m3 = align_vectors((s['p1'] - s['p0']) / height, (0, 1, 0))
    # m2 = translation_matrix((0, .5 * height, 0))
    # m1 = _scale_matrix((s['radius'], .5 * height, s['radius']), enforce_uniform=False)
    #
    # def get_p0(m):
    #     return (m @ np.asarray([0, -1, 0, 1])[:, None])[:3, 0]
    #
    # def get_p1(m):
    #     return (m @ np.asarray([0, 1, 0, 1])[:, None])[:3, 0]
    #
    # p0_ref = s['p0']
    # p0 = get_p0(ret)
    # p1_ref = s['p1']
    # p1 = get_p1(ret)
    # if not np.allclose(p0_ref, p0, atol=1e-3):
    #     import ipdb; ipdb.set_trace()
    # if not np.allclose(p1_ref, p1, atol=1e-3):
    #     import ipdb; ipdb.set_trace()

    # ret will be applied to a Mitsuba cube with corners (-1, -1, -1) and (1, 1, 1)
    return ret


def create_nodes(roots: Optional[list[str]] = None) -> dict[str, Hole]:
    # first create nodes with no edges
    library_equiv: dict[str, Hole] = {}
    while len(library_equiv) < len(library):
        for name in list(library.keys()):
            if name in library_equiv:
                continue
            node = Hole(name=name, docstring=library[name]['docstring'], check=library[name]['check'], normalize=False)
            # manually implement the function
            node.fn = library[name]['__target__']
            library_equiv[name] = node

            # no need to call node because we assume all functions are registered globally
            # try:
            #     _ = node()
            # except Exception as e:
            #     # FIXME give up for now
            #     print(e)
            #     print(traceback.format_exc())
    # print(library_equiv)

    if False: #program_path is not None:
        program = load_program(program_path)
        tree = ast.parse(program)

        def find_calls(func_name):
            class FuncCallVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.calls: list[ast.Call] = []

                def visit_Call(self, call: ast.Call):
                    # Check for direct call to func_name
                    # print('visiting call', call.func.id)
                    if isinstance(call.func, ast.Name) and call.func.id == func_name:
                        self.calls.append(call)
                    # Check for call to library_call with func_name as the first argument
                    elif (isinstance(call.func, ast.Name) and call.func.id == 'library_call' and
                          len(call.args) > 0 and isinstance(call.args[0], ast.Constant) and
                          call.args[0].value == func_name):
                        self.calls.append(call)
                    self.generic_visit(call)

            visitor = FuncCallVisitor()
            visitor.visit(tree)
            return visitor.calls

        def extract_args(call: ast.Call):
            args = []
            for arg in call.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.Tuple):
                    args.append(tuple(el.value for el in arg.elts))
            kwargs = {}
            for kw in call.keywords:
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                elif isinstance(kw.value, ast.Tuple):
                    kwargs[kw.arg] = tuple(el.value for el in kw.value.elts)
            return args, kwargs

        def invoke_calls(func_name):
            calls = find_calls(func_name)
            print(f'[INFO] finding # calls for {func_name}: {len(calls)}')
            success = False
            for call in calls:
                args, kwargs = extract_args(call)
                print(f'[INFO] calling: {func_name}({args=}, {kwargs=})')
                try:
                    _ = library_equiv[func_name](*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
            return success

    else:
        # run root function once to setup `last_call`
        if roots is None:
            print(f'[WARNING] no roots specified, using the last function in the library')
            if ENGINE_MODE == 'mi_from_minecraft':  # FIXME make it unified? also need to fix `optimize_helper.py`
                roots = [next(iter(library.keys()))]
            else:
                roots = [next(reversed(library.keys()))]
        else:
            if len(roots) != 1:
                print(f'[ERROR] number of roots {len(roots)} != 1, {roots=}')
            # assert len(roots) == 1, roots
        with set_track_history_enabled(True):
            for root in roots:
                print(f'[INFO] calling node (supposed to be root): {root}')
                _ = library_call(root)
        hist_calls = {func_name: library[func_name]['hist_calls'].copy() for func_name in library.keys()}  # freeze the list of `hist_calls`
        clear_history()

        def invoke_calls(func_name):
            if len(hist_calls[func_name]) == 0:
                print(f'[INFO] {func_name=} has no last call')
            print(f'[INFO] registering children for {func_name} with {len(hist_calls[func_name])} hist calls')
            for call in hist_calls[func_name]:
                # print(f'[INFO] registering children for {func_name} {call[0]} {call[1]}')
                _ = library_equiv[func_name](*call[0], **call[1])

    # register edges
    for name, node in library_equiv.items():
        if node.children is not None:
            print(f'[INFO] {name=} already has children')
            continue
        with set_fake_call_enabled(True) as _children:
            _children.clear()
            invoke_calls(name)
            child_names = _children.copy()
            _children.clear()
        # manually record the dependency as the program won't call `create_hole`
        node.children = set()
        for child_name in child_names:
            if child_name not in library_equiv:
                # hack
                print(f"[ERROR] {child_name=} not in library_equiv, trying to find an alternative")
                for alt_func_name in library.keys():
                    if library[alt_func_name]['docstring'] == child_name:
                        print(f"[ERROR] {child_name=} not in library_equiv but found an alternative: {alt_func_name=}")
                        child_name = alt_func_name
                        break
            child_node = library_equiv[child_name]
            node.children.add(child_node)
            child_node.add_parent(node)
    return library_equiv


def get_library_source() -> dict[str, str]:
    library_source: dict[str, str] = {}
    for name in library.keys():
        try:
            library_source[name] = inspect.getsource(library[name]['__target__'])
        except TypeError as e:
            print(e)

    return library_source


@contextmanager
def redirect_logs(file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(file_path, 'w') as file:
        class StreamCapture:
            def __init__(self, stream):
                self.stream = stream
                self.file = file

            def write(self, message):
                self.stream.write(message)
                self.file.write(message)

            def flush(self):
                self.stream.flush()
                self.file.flush()

        sys.stdout = StreamCapture(original_stdout)
        sys.stderr = StreamCapture(original_stderr)

        try:
            yield
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    path = 'debug_impl_utils.txt'
    with redirect_logs(path):
        print('debug print')
        raise ValueError('debug error')
