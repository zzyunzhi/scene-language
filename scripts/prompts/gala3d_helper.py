import json
import torch
from typing import Any
from tu.loggers.utils import print_vcv_url
from engine.utils.argparse_utils import modify_string_for_file
from jaxtyping import Float
import cv2
import shlex
import random
import numpy as np
from torchvision.utils import save_image
import sys
from pathlib import Path
from engine.utils.execute_utils import execute_command
from engine.utils.argparse_utils import setup_save_dir
from engine.constants import PROJ_DIR, DEBUG
import os

gala3d_root = Path(PROJ_DIR) / 'engine/third_party/gala3d'

sys.path.insert(0, gala3d_root.as_posix())
from engine.third_party.gala3d.cam_utils import orbit_camera, OrbitCamera, look_at
from engine.third_party.gala3d.gs_renderer import Renderer, MiniCam
from engine.third_party.gala3d.main import GUI
from omegaconf import OmegaConf
from tu.trainers.simple_trainer import load_checkpoint
from tu.utils.pose import assemble_rot_trans_np

root = Path(__file__).parent.parent

# mitsuba3 = opengl: z-backward, y-up, x-right
# colmap:            z-forward, y-down, x-right
# opengl -> colmap
# (1, 0, 0) -> (1, 0, 0)
# (0, 1, 0) -> (0, -1, 0)
# (0, 0, 1) -> (0, 0, -1)


OPENGL_TO_COLMAP = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
COLMAP_TO_OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)


def render_and_save(gui, cam, save_path, mode='rgb'):
    gui.update_parent()
    with torch.no_grad():
        cur_out = gui.renderer.render(cam, gs_model=gui.renderer.gaussians)
    if mode == 'rgb':
        cur_out = cur_out['image']
    elif mode == 'rgba':
        cur_out = torch.cat([cur_out['image'], cur_out['alpha']], dim=0)
    else:
        raise NotImplementedError(mode)
    save_image(cur_out, save_path)


def load_pipe(load_dir: str):
    load_dir = Path(load_dir)
    checkpoint_path = list(sorted(load_dir.glob('*/main/checkpoints/latest.pt'), key=os.path.getmtime))
    if len(checkpoint_path) == 0:
        print(f'[INFO] no checkpoint found from {load_dir}')
        return None
    checkpoint_path = checkpoint_path[-1]
    return load_pipe_from_path(checkpoint_path)


def load_pipe_from_path(checkpoint_path: Path):
    logdir = checkpoint_path.parent.parent
    opt = OmegaConf.load((logdir / 'cfg.json').as_posix())

    opt.outdir = Path(PROJ_DIR) / 'tmp'  # redirect outputs to avoid unintended overwriting
    gui = GUI(opt)
    step = load_checkpoint(gui, opt, checkpoint_path.as_posix())
    print(f'[INFO] loading from step: {step}')

    gui.renderer.gaussians.child = gui.collapse_info['cano_children'] + gui.collapse_info['repl_children']
    # always render interior
    gui.renderer.gaussians.child = [chi for chi in gui.renderer.gaussians.child if not chi.is_exterior]
    gui.update_parent()
    return gui


def run_pipe_post(
        load_dir: str,
        save_dir: str,
        resolution: int,
        sensors: dict[str, dict[str, Any]],
        overwrite: bool = False,
):
    save_dir = Path(save_dir)
    save_frame_paths = [save_dir / f'{sensor_key}_out.png' for sensor_key in sensors.keys()]
    if not overwrite and all([p.exists() for p in save_frame_paths]):
        print(f'[INFO] all frames exist in {save_dir}, skipping')
        return

    gui = load_pipe(load_dir)
    if gui is None:
        return
    opt = gui.opt
    o2w = assemble_rot_trans_np(
        np.asarray(opt.ori[0]) * np.asarray(opt.edge[0]) * np.asarray(opt.scale_factor[0]) / 2,
        np.asarray(opt.center[0])
    ).astype(np.float32)

    for sensor_key, sensor in sensors.items():
        # w'2w = o2w @ w'2o, w' is gala3d world, w is mitsuba sensor world
        trans = o2w @ np.linalg.inv(sensor['o2w'])
        eye = sensor['eye']
        target = sensor['target']
        eye = trans[:3, :3] @ eye + trans[:3, 3]
        target = trans[:3, :3] @ target + trans[:3, 3]
        c2w = assemble_rot_trans_np(look_at(campos=eye, target=target, opengl=True), eye)

        cur_cam = MiniCam(
            c2w.astype(np.float32),
            resolution,
            resolution,
            np.deg2rad(sensor['fov']),
            np.deg2rad(sensor['fov']),
            gui.cam.near,
            gui.cam.far,
        )
        render_and_save(gui, cur_cam, (save_dir / f'{sensor_key}_out.png').as_posix())


def run_pipe_post_animation(
        frames: list[list[Float[np.ndarray, "4 4"]]],
        load_dir: str,
        save_dir: str,
        resolution: int,
        cam_radius: float,
):
    gui = load_pipe(load_dir)
    if gui is None:
        return
    cam = MiniCam(
        orbit_camera(-20, 0, cam_radius),
        resolution,
        resolution,
        gui.cam.fovy,
        gui.cam.fovx,
        gui.cam.near,
        gui.cam.far,
    )

    gui.renderer.gaussians.child = gui.collapse_info['cano_children'] + gui.collapse_info['repl_children']
    eids = gui.collapse_info['cano_eids'] + gui.collapse_info['repl_eids']
    init_object_centers = gui.collapse_info['cano_centers'] + gui.collapse_info['repl_centers']
    init_scale_factors = gui.collapse_info['cano_scale_factors'] + gui.collapse_info['repl_scale_factors']
    cid_to_arrays = {}
    for cid in range(len(gui.renderer.gaussians.child)):
        child = gui.renderer.gaussians.child[cid]
        cid_to_arrays[cid] = {
            'object_center': child.object_center.clone().detach().cpu().numpy(),
            'scale_factor': child.scale_factor.clone().detach().cpu().numpy(),
            'ori': child.ori.copy(),
        }

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for frame_ind, toworld_list in enumerate(frames):
        toworld_dict = parse_toworld(toworld_list)
        centers = toworld_dict['centers']
        edges = toworld_dict['edges']
        scale_factors = toworld_dict['scale_factors']
        rotations = toworld_dict['rotations']

        for cid in range(len(gui.renderer.gaussians.child)):
            eid = eids[cid]
            rel_center = np.asarray(centers[eid]) + cid_to_arrays[cid]['object_center'] - init_object_centers[cid]
            rel_scale_factor = np.asarray(scale_factors[eid]) * cid_to_arrays[cid]['scale_factor'] / init_scale_factors[cid]
            rel_rotation = np.asarray(rotations[eid]) #  @ eid_to_arrays[eid]['ori']

            child = gui.renderer.gaussians.child[cid]
            child.object_center = torch.tensor(rel_center, dtype=torch.float32, device="cuda")
            child.scale_factor = torch.tensor(rel_scale_factor, dtype=torch.float32, device="cuda")
            child.ori = rel_rotation

        render_and_save(gui, cam, (save_dir / f'{frame_ind:03d}.png').as_posix())


def parse_toworld(toworld_list: list[Float[np.ndarray, "4 4"]]):
    centers: list[Float[np.ndarray, "3"]] = []
    edges: list[Float[np.ndarray, "3"]] = []
    scale_factors: list[float] = []
    rotations: list[Float[np.ndarray, "3 3"]] = []
    for mat in toworld_list:
        mat = mat.astype(np.float32)
        # mat = OPENGL_TO_COLMAP @ mat @ COLMAP_TO_OPENGL
        center = mat[:3, 3]  # (3,)
        scale = np.linalg.norm(mat[:3, :3], axis=0)  # (3,)
        rotation = mat[:3, :3] / scale  # (3, 3)
        centers.append(center)
        edge = scale * 2  # 2 is mitsuba cube edge size with identity pose
        scale_factor = np.linalg.norm(edge)
        edges.append(edge / scale_factor)
        scale_factors.append(scale_factor)
        rotations.append(rotation)

    return {
        'centers': centers,
        'edges': edges,
        'scale_factors': scale_factors,
        'rotations': rotations,
    }


def run_pipe(toworld_list: list[Float[np.ndarray, "4 4"]],
             docstrings: list[str],
             negative_docstrings: list[str],
             exterior_flags: list[bool],
             yaws: list[float],
             prompt: str,
             save_dir: str, resolution: int,
             cam_radius: float,
             sensors: dict[str, dict[str, Any]],
             overwrite: bool = False,
             ):
    save_dir = Path(save_dir)
    checkpoint_path = list(sorted(save_dir.glob('*/main/checkpoints/latest.pt'), key=os.path.getmtime))

    if not overwrite and len(checkpoint_path) > 0:
        return

    toworld_dict = parse_toworld(toworld_list)
    centers = toworld_dict['centers']
    edges = toworld_dict['edges']
    scale_factors = toworld_dict['scale_factors']
    rotations = toworld_dict['rotations']

    cam = OrbitCamera(resolution, resolution, r=cam_radius, fovy=49.1)
    renderer = Renderer(sh_degree=0, centers=centers, edges=edges, scale_factors=scale_factors, floor=True)
    renderer.initialize(num_pts=100000)  # 100000

    for i in range(len(docstrings)):
        renderer.gaussians.child[i].prompt = docstrings[i]
        renderer.gaussians.child[i].ori = rotations[i]
        renderer.gaussians.child[i].is_exterior = exterior_flags[i]

    # only render interior
    # TODO render exterior too
    renderer.gaussians.child = [chi for chi in renderer.gaussians.child if not chi.is_exterior]

    renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in renderer.gaussians.child], dim=0)
    renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in renderer.gaussians.child], dim=0)
    renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in renderer.gaussians.child], dim=0)
    renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in renderer.gaussians.child], dim=0)
    renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in renderer.gaussians.child], dim=0)
    renderer.gaussians._opacity = torch.cat([chi._opacity for chi in renderer.gaussians.child], dim=0)

    o2w = toworld_list[0]
    gs_init_save_dir = save_dir / 'gs_init'
    gs_init_save_dir.mkdir(exist_ok=True, parents=True)
    for sensor_key, sensor in sensors.items():
        eye = sensor['eye']
        target = sensor['target']

        # w'2w = o2w @ w'2o, w' is gala3d world, w is mitsuba sensor world
        trans = o2w @ np.linalg.inv(sensor['o2w'])
        eye = trans[:3, :3] @ eye + trans[:3, 3]
        target = trans[:3, :3] @ target + trans[:3, 3]
        c2w = assemble_rot_trans_np(look_at(campos=eye, target=target, opengl=True), eye)

        # # not sure why the following is wrong... the camera target becomes incorrect (shifted along y)
        # c2w = trans @ sensor['c2w']
        # c2w[:3, :3] = c2w[:3, :3] / np.linalg.norm(c2w[:3, :3], axis=0)
        # c2w[0, :3] *= -1
        # c2w[2, :3] *= -1
        #
        # print('ref', np.round(c2w1, 3))
        # print(c2w1[:3, :3] @ c2w1[:3, :3].T)
        # print('actual', np.round(c2w, 3))
        # print(c2w[:3, :3] @ c2w[:3, :3].T)
        cur_cam = MiniCam(
            c2w.astype(np.float32),
            resolution,
            resolution,
            np.deg2rad(sensor['fov']),
            np.deg2rad(sensor['fov']),
            cam.near,
            cam.far,
        )

    # vers = [-30] * 36
    # hors = [i * 10 for i in range(-18, 18)]
    # render_resolution = resolution
    # for i in range(36):
    #     c2w = orbit_camera(vers[i], hors[i], cam_radius)
    #     cur_cam = MiniCam(
    #         c2w,
    #         render_resolution,
    #         render_resolution,
    #         cam.fovy,
    #         cam.fovx,
    #         cam.near,
    #         cam.far,
    #     )

        with torch.no_grad():
            cur_out = renderer.render(cur_cam, gs_model=renderer.gaussians)["image"]
        input_tensor = cur_out.detach().cpu().numpy() * 255.0
        input_tensor = input_tensor.transpose(1, 2, 0)
        input_tensor = input_tensor.astype(np.uint8)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        # cv2.imwrite((gs_init_save_dir / f'{i:03d}.png').as_posix(), input_tensor)
        cv2.imwrite((gs_init_save_dir / f'{sensor_key}.png').as_posix(), input_tensor)

    if not overwrite and len(checkpoint_path) > 0:
        return

    # use existing checkpoint instead of rerunning even with overwrite = True
    # FIXME manually comment it out to rerun
    # if len(checkpoint_path) > 0:
    #     return

    base_info = {
        'center': np.stack(centers).tolist(),
        'edge': [[round(float(f), 3) for f in edge] for edge in edges],
        'scale_factor': [float(f) for f in scale_factors],
        'ori': np.stack(rotations).tolist(),
        'yaw': [round(float(yaw), 2) for yaw in yaws],  # in degrees
        'prompt': docstrings,
        'negative_prompt': negative_docstrings,
        'radius': cam_radius,
        'floor': None,
        'scene': prompt,
        'collapse_prompts': True,
        'seed': 0,
        'load_objects_w_edit': False,
        'load_objects_edits': None,
        'exterior_flags': exterior_flags,
        'exterior_prob': 0, # 0.2,  # FIXME
        # 'elevation': -10,  # TODO
        # 'use_densify_and_prune': True,  # TODO
        # 'interval': 99999999,  # TODO
        # 'adjust_num_pts': True,  # TODO
        # 'scene_image': "/viscam/projects/concepts/zzli/images/paintings/Wayne_Thiebaud_Confections_1962.png",
        # 'scene_image': "/viscam/projects/concepts/zzli/images/paintings/Ceramic_Vases.png",
        # 'scene_image': "/viscam/projects/concepts/zzli/images/paintings/still_life_skull_anthony_nold.png",
    }
    run_style_transfer = False  # False
    if run_style_transfer:
        base_info.update({
            # 'scene_image': "/viscam/projects/concepts/engine/scripts/exp/icl_0512/submission/outputs/identity/coke_bottle_2bad8af7-9ca9-59c2-aa92-ae15c89d8e15_max_it_200_lr_0.004/info.pth",
            # 'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/balloon_dog/dog_c9615669-181c-5685-8551-fb3e0da55933_max_it_40_lr_0.004/info.pth",
            # 'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/wood/board_a7ee2412-a2d4-59c6-9000-6f0dc52df3ea_max_it_40_lr_0.004/info.pth",
            # "scene_image": "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/moai/moai_6a0bdee9-c379-5195-945a-2d38c3aa1887_max_it_40_lr_0.004/info.pth",
            # "scene_image": "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/colorful-moai/moai_6a0bdee9-c379-5195-945a-2d38c3aa1887_max_it_20_lr_0.004/info.pth",
            "scene_image": "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/moai/moai_6a0bdee9-c379-5195-945a-2d38c3aa1887_max_it_100_lr_0.004_opt_cls_False/info.pth",
            'templates': ['{cls_token}, {color_token}'],
            'load_color_token': True,
            'load_special_token': True,
            'load_cls_token': False,
            # "negative_prompt": ['unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation.'] * len(negative_docstrings)
        })

    override_kwargs_list = [
        # {'densify_from_iter': 50, 'densify_until_iter': 5000, 'densification_interval': 1000, 'percent_dense': 0.01, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005},
        # ('base', None),
        # percent_dense = 0 -> no clone, only split; percent_dense ++ -> more clone, less split
        # densify_grad_threshold small -> more clone and split in total; large -> less clone and split in total
        # min_opac_prune small: less pruning; large -> more pruning

        # only clone, no split

        # # higher grad thresh
        # ('clone_only_high_grad_tresh', {'densify_from_iter': 50, 'densify_until_iter': 5000, 'densification_interval': 1000, 'percent_dense': 0.01, 'densify_grad_threshold': 0.001, 'min_opac_prune': 0.005}),
        # # longer densification duration
        # ('clone_only_long_dsf_duration', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 2000, 'percent_dense': 0.01, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005}),
        # # less frequent densification
        # ('clone_only_less_freq_dsf', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 4000, 'percent_dense': 0.01, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005}),
        # # more frequent densification  # OOM
        # ('clone_only_more_freq_dsf', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 500, 'percent_dense': 0.01, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005}),

        # no clone, only split

        # # higher grad thresh
        # ('split_only_high_grad_tresh', {'densify_from_iter': 50, 'densify_until_iter': 5000, 'densification_interval': 1000, 'percent_dense': 0, 'densify_grad_threshold': 0.001, 'min_opac_prune': 0.005}),
        # # longer densification duration
        # ('split_only_long_dsf_duration', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 2000, 'percent_dense': 0, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005}),
        # # less frequent densification
        # ('split_only_less_freq_dsf', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 4000, 'percent_dense': 0, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005}),
        # # more frequent densification  # OOM
        # ('split_only_more_freq_dsf', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 500, 'percent_dense': 0, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.005}),

        # ('clone_only_more_freq_dsf_high_opac_thresh', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 1000, 'percent_dense': 0.01, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.02, 'lambda_layout': 10, 'feature_lr': 0.002}),
        # ('split_only_more_freq_dsf_high_opac_thresh', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 1000, 'percent_dense': 0, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.02, 'lambda_layout': 10, 'feature_lr': 0.002}),
        # ('clone_only_more_freq_dsf_high_opac_thresh_less_reg', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 1000, 'percent_dense': 0.01, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.02, 'lambda_layout': 10, 'feature_lr': 0.002, 'lambda_reg': 100}),
        # ('split_only_more_freq_dsf_high_opac_thresh_less_reg', {'densify_from_iter': 50, 'densify_until_iter': 10000, 'densification_interval': 1000, 'percent_dense': 0, 'densify_grad_threshold': 0.0002, 'min_opac_prune': 0.02, 'lambda_layout': 10, 'feature_lr': 0.002, 'lambda_reg': 100}),
        # ('guidance_10', {'sd_guidance_scale': 10, 'cldm_guidance_scale': 10}),

        # ('base', None),
        # ('it100', {'sdti.max_it': 100}),
        # ('it100guidance100', {'sdti.max_it': 100, "mvdream_guidance_scale": 100}),
        # ('colorful-moai', {'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/colorful-moai"}),
        # ('colorful-moai-it100', {'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/colorful-moai", 'sdti.max_it': 100}),
        # ('colorful-moai-it100guidance100', {'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/colorful-moai", 'sdti.max_it': 100, "mvdream_guidance_scale": 100}),
        # ('purple-moai', {'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/purple-moai"}),
        # ('purple-moai-it100', {'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/purple-moai", 'sdti.max_it': 100}),
        # ('purple-moai-it100guidance100', {'scene_image': "/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/purple-moai", 'sdti.max_it': 100, "mvdream_guidance_scale": 100}),
    ]

    # override_kwargs_list = []
    # for scene_image in Path("/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/texture/").glob("*"):
    #     override_kwargs_list.append((scene_image.stem, {'scene_image': scene_image.as_posix()}))

    # override_kwargs_list = []
    # style_root = Path("/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style")
    # for color_token, filename in [
    #     # ('silver', 'texture/teapot.png'),
    #     # ('china', 'texture/vase.png'),
    #     # ('colorful', 'texture/mask.png'),
    #     # ('colorful', 'texture/egg.png'),
    #     # ('African black and white', 'texture/face.png'),
    #     # ('gothic', 'texture/window.png'),
    #     # ('gothic glass', 'texture/window.png'),
    # ]:
        # override_kwargs_list.append((Path(filename).stem, {'scene_image': (style_root / filename).as_posix(), 'sdti.color_token': color_token}))
        # override_kwargs_list.append(('it200_' + Path(filename).stem, {'scene_image': (style_root / filename).as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 200}))
        # override_kwargs_list.append(('it200_cfg100' + Path(filename).stem, {'scene_image': (style_root / filename).as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 200, 'mvdream_guidance_scale': 100}))
        # override_kwargs_list.append(('it100_' + Path(filename).stem, {'scene_image': (style_root / filename).as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 100}))
        # override_kwargs_list.append(('it100_cfg100' + Path(filename).stem, {'scene_image': (style_root / filename).as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 100, 'mvdream_guidance_scale': 100}))
    override_kwargs_list = []
    for scene_image in Path("/viscam/projects/concepts/engine/engine/third_party/gala3d/assets/style/material/").glob("vase__raw_clay.png"):
        phrases = scene_image.stem.split('__')
        _, color_token, *_ = phrases
        color_token = color_token.replace('_', ' ')
        override_kwargs_list.append((scene_image.stem + 'it10', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 10}))
        override_kwargs_list.append((scene_image.stem + 'it10_cfg100', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 10, 'mvdream_guidance_scale': 100}))
        override_kwargs_list.append((scene_image.stem + 'it20', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 20}))
        override_kwargs_list.append((scene_image.stem + 'it20_cfg100', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 20, 'mvdream_guidance_scale': 100}))
        # override_kwargs_list.append((scene_image.stem + 'it50', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 50}))
        # override_kwargs_list.append((scene_image.stem + 'it50_cfg100', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 50, 'mvdream_guidance_scale': 100}))
        # override_kwargs_list.append((scene_image.stem + 'it100_', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 100}))
        # override_kwargs_list.append((scene_image.stem + 'it100_cfg100', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 100, 'mvdream_guidance_scale': 100}))
        # override_kwargs_list.append((scene_image.stem + 'it200_', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 200}))
        # override_kwargs_list.append((scene_image.stem + 'it200_cfg100', {'scene_image': scene_image.as_posix(), 'sdti.color_token': shlex.quote(color_token), 'sdti.max_it': 200, 'mvdream_guidance_scale': 100}))

    for extra_config in [
        # 'adjust_true_densify_true_elev_n10',
        # 'adjust_true_densify_true_elev_n10_no_layout',
        # 'adjust_true_densify_true_elev_n10_no_reg',
        # 'adjust_true_densify_true_elev_n10_adjust_cameras_extent_true',
        # 'adjust_true_densify_true_elev_n10_adjust_cameras_extent_true_layout_100',
        # 'adjust_true_densify_true_elev_n10_adjust_cameras_extent_true_layout_100_ft_lr_1en3',
        # 'pct_dense_1en2_grad_thresh_2en4',
        # 'pct_dense_1en2_grad_thresh_1en3',
        # 'pct_dense_0_grad_thresh_1en3',
        # 'pct_dense_0_grad_thresh_2en4',
        # 'elev0',
        # 'elev0_no_df',
        # 'elev0_df_once',
        # 'elev0_df_once_layout_20',
        # 'elev0_no_df_layout_20',
        # 'elev0_df_once_split_only',
        # 'elev0_df_once_split_only_layout_20',
        # 'elev0_no_df',
        # 'elev0_no_df_layout_1000',
        # 'style_moai',
        # 'style_transfer',
        'style_transfer_material',
    ]:
        for exp_name, override_kwargs in override_kwargs_list:
            exp_name = extra_config + '___' + exp_name
            launch_slurm(save_dir=save_dir, base_info=base_info, exp_name=exp_name, extra_configs=[extra_config], override_kwargs=override_kwargs)


def launch_slurm(save_dir: Path, base_info: dict, exp_name: str, extra_configs: list = (), override_kwargs: dict = None):
    # exp_name = f'gala3d_helper_{"___".join(extra_configs)}'
    # if override_kwargs is not None:
    #     exp_name += '___' + modify_string_for_file("__".join([f"{k}_{v}" for k, v in override_kwargs.items()]), append_uuid=False)
    print(f'[INFO] launching {exp_name}')
    exp_dir = setup_save_dir((save_dir / exp_name).absolute().as_posix(), log_unique=True)
    # if len(centers) > 100:
    #     # too many objects, disable CLDM
    #     info.update({'interval': 0})

    info = {**base_info, 'outdir': (exp_dir / 'main').as_posix()}

    info_save_path = exp_dir / 'info.json'

    with open(info_save_path.as_posix(), 'w') as f:
        json.dump(info, f, indent=4)

    print_vcv_url(info_save_path.as_posix())

    base_configs = [root / 'configs' / (config_name + '.yaml') for config_name in ['bedroom'] + extra_configs]
    for base_config in base_configs:
        if not base_config.exists():
            raise FileNotFoundError(f'base config {base_config} not found')
    configs = [base_config.as_posix() for base_config in base_configs] + [info_save_path.as_posix()]
    local_command = ['python', (gala3d_root / 'main.py').as_posix(), '--configs', *configs]
    if override_kwargs is not None:
        local_command.extend(['--', ' '.join([f'{k}={v}' for k, v in override_kwargs.items()])])

    '''
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/outputs/assets/cokes/gala3d_two_coke_can_stacks_indoors_no_window_depth_02_frame_00/gala3d_helper_20240817-011927_e073f1eb-0ec1-482d-8934-82ee701f38a3/info.json
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/outputs/assets/cokes/gala3d_two_coke_can_stacks_indoors_no_window_depth_02_frame_00/gala3d_helper_20240817-011927_e073f1eb-0ec1-482d-8934-82ee701f38a3/info_edit.json
    for the coffee table scene:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/outputs/assets/coffee_table/gala3d_scene_indoors_no_window_depth_02_frame_00/gala3d_helper_20240828-083359_97ad8ec3-e0da-4e49-81df-7f1f702ddab3/info.json

    for the cokes scene:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/outputs/assets/cokes/gala3d_two_coke_can_stacks_indoors_no_window_depth_02_frame_00/gala3d_helper_20240823-230005_5b60c398-e359-437e-bf17-20b336d03bbc/info.json
    
    for debug cokes2 scene:
    ENGINE_MODE=mi DEBUG=0 PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/engine/scripts/exp/icl_0512/outputs/render/edit/debug/coke2/impl.py --engine-modes gala3d --log-dir /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/outputs/assets/coke2 --program-path /viscam/projects/concepts/engine/scripts/exp/icl_0512/assets/edit/debug/coke2/program.py
    
    for 6 soda can (coke_s) scene:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/outputs/assets/coke_s/gala3d_coca_cola_pack_indoors_no_window_depth_02_frame_00/gala3d_helper_20240828-182525_e2395905-bf1a-41b8-8982-0aa3f9a12771/info.json

    for Wayne_Thiebaud_Confections_1962 depth=1, SAM results need to check
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/assets/outputs/assets/Wayne_Thiebaud_Confections_1962/gala3d_dessert_scene_indoors_no_window_depth_01_frame_00/gala3d_helper_20240907-192605_6b3529b2-3deb-45e2-923f-04e801d4ff4d/info.json
    
    for Ceramic Vases scene depth=1, SAM results need to check
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/assets/outputs/assets/Ceramic_Vases/gala3d_scene_indoors_no_window_depth_01_frame_00/gala3d_helper_20240908-012110_ed79f0ac-1f75-48d7-b4e8-62f0b7406b3e/info.json

    for skull scene depth=1:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/assets/outputs/assets/still_life_skull_anthony_nold/gala3d_scene_indoors_no_window_depth_01_frame_00/gala3d_helper_20240907-194058_9abce0be-3e61-4bcc-822b-6a1814fc6007/info.json
    
    for skull scene edited:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/assets/outputs/assets/still_life_skull_anthony_nold_edit/gala3d_scene_indoors_no_window_depth_-1_frame_00/gala3d_helper_20240914-175814_7da37e15-13ed-4226-94a8-30004f6ad4dc/info.json
    
    for Ceramic Vases scene edited:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/assets/outputs/assets/Ceramic_Vases_edit/gala3d_scene_indoors_no_window_depth_-1_frame_00/gala3d_helper_20240914-174926_d7171376-949a-46e0-917d-25a1f2bf7c02/info.json
    
    for skull scene edited new:
    PYTHONPATH=/viscam/projects/concepts/engine/engine/third_party/gala3d:/viscam/projects/concepts/third_party:/viscam/projects/concepts/zzli/engine:/viscam/u/yzzhang/projects/tu:/viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/prompts:$PYTHONPATH python /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/main.py --configs /viscam/projects/concepts/zzli/engine/engine/third_party/gala3d/configs/bedroom.yaml /viscam/projects/concepts/zzli/engine/scripts/exp/icl_0512/assets/outputs/assets/still_life_skull_anthony_nold_edit/gala3d_scene_indoors_no_window_depth_-1_frame_00/gala3d_helper_20240916-143248_7bee2ff3-7e33-4283-9e7f-5a04fe57a4a7/info.json
    
    '''

    # info_edit = {
    #     **info,
    #     'load_objects_w_edit': True,
    #     # TODO: automate this
    #     'load_objects_edits': {
    #         'coke can': [
    #             [0, 0, 0],
    #             [0.5, 0.5, -1],
    #             [0, 0.5, 0]
    #         ]
    #     }
    # }
    # info_edit_save_path = exp_dir / 'info_edit.json'
    # with open(info_edit_save_path.as_posix(), 'w') as f:
    #     json.dump(info_edit, f, indent=4)
    # print_vcv_url(info_edit_save_path.as_posix())
    # local_edit_command = [
    #     'python', (gala3d_root / 'main.py').as_posix(),
    #     '--configs', (gala3d_root / 'configs' / 'bedroom.yaml').as_posix(),
    #     info_edit_save_path.as_posix()
    # ]
    # print(' '.join(local_command))
    # from pdb import set_trace; set_trace()
    # raise RuntimeError('debugging exit..')
    execute_command(command=' '.join(local_command), save_dir=(exp_dir / 'local').as_posix(), cwd=gala3d_root.as_posix(), dry_run=True)
    # execute_command(command=' '.join(local_edit_command), save_dir=(exp_dir / 'local_edit').as_posix(), cwd=gala3d_root.as_posix(), dry_run=True)
    # exit()

    slurm_command = [
        "python", "-m", "tu.sbatch.sbatch_sweep",
        "--command", shlex.quote(' '.join(local_command)),
        "--cpus_per_task", "32",
        "--mem", "32G",
        "--proj_dir", PROJ_DIR,
        "--console_output_dir", (exp_dir / 'srun').as_posix(),
        "--job", base_info['scene'].replace(' ', '_') + '_' + exp_dir.name,
        "--conda_env", "engine",
        # "exclude=viscam[1-4],viscam10,viscam12,viscam13"
        # "exclude=viscam[1-4],viscam[5-9]"
        "exclude=viscam[1-4],viscam12"
        # "exclude=viscam[1-4],viscam[6-8],viscam10,viscam[11-13]"
        # "exclude=viscam[1-4],viscam[6-8],viscam12"
        # '--conda_env', "llama",
        # "exclude=viscam[1-10]"
    ]
    execute_command(command=' '.join(slurm_command), save_dir=(exp_dir / 'sbatch').as_posix(), cwd=PROJ_DIR, dry_run=False,
                    print_stderr=True, print_stdout=True)


# python scripts/render.py --input-dir assets/edit/ --input-pattern "paintings/*Castle*/*" "pain*/*Son*/*" "layout/*three_coke*/*" "scul*/*Linco*/*" --overwrite