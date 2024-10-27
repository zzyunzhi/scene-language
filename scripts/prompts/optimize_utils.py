import drjit as dr
import mitsuba as mi

import numpy as np
import numpy.typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import uuid
from PIL import Image
from pathlib import Path
from contextlib import redirect_stdout
from tqdm import tqdm
import os

from engine.utils.mitsuba_utils import compute_bbox as _compute_bbox
from mi_helper import _preprocess_shape, concatenate_xml_files, suppress_output, fov_from_box_cam
from math_utils import _scale_matrix, translation_matrix, rotation_matrix
# from threestudio.prompt_processor import StableDiffusionPromptProcessor
# from threestudio.guidance import StableDiffusionGuidance

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

device = torch.device('cuda')
SPP = 2048 #128 #4096


# pytorch3d

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# mitsuba3
@dr.wrap_ad(source='torch', target='drjit')
def render_scene_w_pose(
    scene,
    rotation_rep='axis-aligned',
    trans=None,
    rot_x=None,
    rot_y=None,
    rot_z=None,
    mtx=None,
    cam=None,
    spp=256,
    seed=1
):
    sensor = cam
    integrator = mi.load_dict(
        {
            'type': 'direct_projective',
        }
    )
    if rotation_rep == 'quaternion':
        raise NotImplementedError
    
    elif rotation_rep == 'axis-aligned':
        num_shapes = trans.shape[0]

        params = mi.traverse(scene)
        for i in range(num_shapes):
            initial_vertex_positions = dr.unravel(mi.Point3f, params[f'{i:02d}.vertex_positions'])
            
            # dr.grad_enabled(translate) check the gradient in mitsuba
            # dr.enable_grad(trans)
            # dr.enable_grad(rot_y)
            translate = mi.Point3f(trans[i,0].array, trans[i,1].array, trans[i,2].array)
            print(f'gradient of translate is: ', dr.grad(translate))
            rotate_y = mi.Float32(rot_y[i,0].array)

            # dr.enable_grad(translate)
            # dr.enable_grad(rotate_y)
            print(f'gradient of translate is: ', dr.grad(translate))

            trafo = mi.Transform4f.translate(translate).rotate([0, 1, 0], rotate_y * 100.0)
            params[f'{i:02d}.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
            # dr.enable_grad(params[f'{i:02d}.vertex_positions'])
        
        params.update()
        # break
    
    image = mi.render(scene, params=params, sensor=sensor, integrator=integrator, spp=spp, seed=seed, seed_grad=seed+1)
    # image = mi.render(scene, params=params, sensor=sensor, spp=spp, seed=seed, seed_grad=seed+1)
    print('gradient of mi render image is: ', dr.grad(image).array)
    return image, scene


@dr.wrap_ad(source='drjit', target='torch')
def sds_guidance(img, model_prompt_processor, model_guidance, sample_cam):
    prompt_utils = model_prompt_processor.__call__()
    guidance_out = model_guidance.__call__(
        img.unsqueeze(0), prompt_utils,
        elevation=sample_cam['elevation_deg'],
        azimuth=sample_cam['azimuth_deg'],
        camera_distances=sample_cam['camera_distances'],
        rgb_as_latents=False,
        guidance_eval=False,
        save_guidance_eval_utils=True
    )

    loss = guidance_out['loss_sds']
    return loss

# threestudio
# model_prompt_processor = None
# model_guidance = None
# def try_setup_sd(prompt):
#     global model_prompt_processor, model_guidance
#     prompt = prompt.replace('_', ' ')
#     if model_prompt_processor is not None and model_prompt_processor.prompt == prompt:
#         return

#     model_prompt_processor = StableDiffusionPromptProcessor({'prompt': prompt})
#     model_guidance = StableDiffusionGuidance({
#         'pretrained_model_name_or_path': "stabilityai/stable-diffusion-2-1-base",
#         'guidance_scale': 100.,
#         'min_step_percent': 0.02,
#         'max_step_percent': 0.98,
#     })

def debug_layout_optimize(scene, keys):
    integrator = mi.load_dict({'type': 'direct_projective'})
    params = mi.traverse(scene)
    init_vert_positions = {k: dr.unravel(mi.Point3f, params[f'{k}.vertex_positions']) for k in keys}
    opt = mi.ad.Adam(lr=0.025)
    for k in keys:
        opt[f'{k}_angle'] = mi.Float(.25)
        opt[f'{k}_trans'] = mi.Point3f(.1, -.25, .1)

    def apply_transformation(k: str):
        opt[f'{k}_trans'] = dr.clamp(opt[f'{k}_trans'], -0.5, 0.5)
        opt[f'{k}_angle'] = dr.clamp(opt[f'{k}_angle'], -0.5, 0.5)
        trafo = mi.Transform4f.translate(opt[f'{k}_trans']).rotate([0, 1, 0], opt[f'{k}_angle'] * 100.0)
        params[f'{k}.vertex_positions'] = dr.ravel(trafo @ init_vert_positions[k])
        params.update()

    for it in range(10):
        for k in keys:
            apply_transformation(k)
        img = mi.render(scene, params, seed=it, spp=16, integrator=integrator)

        loss = dr.sum(dr.sqr(img)) / len(img)
        dr.backward(loss)
        opt.step()

        print(loss)
    return scene


def compute_closest_fov(camera_location, box):
    # when looking at [0,0,0] from transitions, compute the fov value
    point = camera_location
    box_points = [
        np.asarray((box.center[0] - box.sizes[0] * .5, box.center[1] - box.sizes[1] * .5, box.center[2] - box.sizes[2] * .5)),
        np.asarray((box.center[0] + box.sizes[0] * .5, box.center[1] - box.sizes[1] * .5, box.center[2] - box.sizes[2] * .5)),
        np.asarray((box.center[0] - box.sizes[0] * .5, box.center[1] + box.sizes[1] * .5, box.center[2] - box.sizes[2] * .5)),
        np.asarray((box.center[0] + box.sizes[0] * .5, box.center[1] + box.sizes[1] * .5, box.center[2] - box.sizes[2] * .5)),
        np.asarray((box.center[0] - box.sizes[0] * .5, box.center[1] - box.sizes[1] * .5, box.center[2] + box.sizes[2] * .5)),
        np.asarray((box.center[0] + box.sizes[0] * .5, box.center[1] - box.sizes[1] * .5, box.center[2] + box.sizes[2] * .5)),
        np.asarray((box.center[0] - box.sizes[0] * .5, box.center[1] + box.sizes[1] * .5, box.center[2] + box.sizes[2] * .5)),
        np.asarray((box.center[0] + box.sizes[0] * .5, box.center[1] + box.sizes[1] * .5, box.center[2] + box.sizes[2] * .5)),
    ]

    fov = fov_from_box_cam(point, box_points, box)
    return fov


def layout_optimize_mi(shape, prompt, save_dir, preset_xml_path, target_box, optimize_iter=1e2):
    prompt = prompt.replace('_', ' ')
    model_prompt_processor = StableDiffusionPromptProcessor({'prompt': prompt})
    model_guidance = StableDiffusionGuidance({
        'pretrained_model_name_or_path': "stabilityai/stable-diffusion-2-1-base",
        'guidance_scale': 10, #100.,
        'min_step_percent': 0.02,
        'max_step_percent': 0.05,
    })

    optimize_iter = int(optimize_iter)
    num_shapes = len(shape)
    shape = _preprocess_shape(shape)
    scene_dict = {'type': 'scene', **{f'{i:02d}': s for i, s in enumerate(shape)}}

    # TODO: remove background things in this scene
    tmp_xml_file = Path(preset_xml_path).with_name(f'tmp_{uuid.uuid4()}.xml')
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        mi.xml.dict_to_xml(scene_dict, tmp_xml_file)
    tree = concatenate_xml_files(preset_xml_path, tmp_xml_file.as_posix())
    tree.write(tmp_xml_file.as_posix())
    with suppress_output():
        scene: mi.Scene = mi.load_file(tmp_xml_file.as_posix())

    LR = 0.025
    LOSS_WEIGHT = 0.001
    keys = [f"{i:02d}" for i in range(num_shapes)]

    params = mi.traverse(scene)
    init_vert_positions = {k: dr.unravel(mi.Point3f, params[f'{k}.vertex_positions']) for k in keys}
    opt = mi.ad.Adam(lr=LR)

    for k in keys:
        # opt[f'{k}_angle'] = mi.Float(.25)
        # opt[f'{k}_trans'] = mi.Point3f(.1, -.25, .1)
        opt[f'{k}_angle'] = mi.Float(.0)
        opt[f'{k}_trans'] = mi.Point3f(.0, .0, .0)
    
    def apply_transformation(k: str):
        opt[f'{k}_trans'] = dr.clamp(opt[f'{k}_trans'], -0.5, 0.5)
        opt[f'{k}_angle'] = dr.clamp(opt[f'{k}_angle'], -0.5, 0.5)
        trafo = mi.Transform4f.translate(opt[f'{k}_trans']).rotate([0, 1, 0], opt[f'{k}_angle'] * 100.0)
        params[f'{k}.vertex_positions'] = dr.ravel(trafo @ init_vert_positions[k])
        params.update()
    
    integrator = mi.load_dict({'type': 'direct_projective'})
    
    bbox = _compute_bbox(scene_dict)
    center = bbox.center
    radius = max(bbox.sizes / 2)

    # only sample once
    sample_cam = random_sample_camera()
    for k,v in sample_cam.items():
        sample_cam[k] = v.to(device)
    
    for it in tqdm(range(optimize_iter), desc="optimizing poses"):
        # sample elevation, azimuth and camera distance for SDS
        # I use the radius as 1.0 in camera distance unit
        # sample_cam = random_sample_camera()
        distance = sample_cam['camera_distances'] * torch.Tensor([radius]).to(device) * 2
        camera_positions = torch.stack(
            # different from threestudio, in mitsuba3 the y-axis facing upwards
            [
                distance * torch.cos(sample_cam['elevation']) * torch.cos(sample_cam['azimuth']),
                distance * torch.sin(sample_cam['elevation']),
                distance * torch.cos(sample_cam['elevation']) * torch.sin(sample_cam['azimuth']),
            ],
            dim=-1,
        ).reshape(-1).cpu().numpy()
        camera_positions = camera_positions + center

        sample_fov = compute_closest_fov(camera_positions, bbox)

        # for k,v in sample_cam.items():
        #     sample_cam[k] = v.to(device)
        
        render_cam: mi.Sensor = mi.load_dict({
            'type': 'perspective',
            # 'to_world': mi.scalar_rgb.Transform4f.translate(
            #     box.size * np.asarray(rel_offset),
            # ) @ canon_sensor.world_transform(),
            'to_world': mi.scalar_rgb.Transform4f.look_at(
                # origin=np.array(canon_transform.translation()) + box.size * np.asarray([0, .4, -2]),
                # origin=np.array([target_box.center[0], target_box.max[1], target_box.max[2]]),
                origin=camera_positions,
                # target=np.array(canon_transform.matrix)[:3, :3] @ np.array([0, 0, 1]) + np.array(canon_transform.translation()),
                target=center,
                up=[0, 1, 0]
            ),
            'fov': sample_fov, #60,
            'film': {
                'type': 'hdrfilm',
                'width': 256,
                'height': 256,
                'sample_border': True,
            }
        })

        for k in keys:
            apply_transformation(k)
        img = mi.render(scene, params, sensor=render_cam, seed=it, spp=SPP, integrator=integrator)
        image_before = mi.util.convert_to_bitmap(img)
        image_before = np.asarray(image_before)
        
        # TODO:
        '''
        1. render albedo with no background
        2. render_ad(): PSIntegrator detected the potential for image-space motion due to differentiable shape parameters. 
            To correctly account for shapes entering or leaving the viewport, it is recommended that you set the film's 'sample_border' parameter to True
        '''

        loss = sds_guidance(img, model_prompt_processor, model_guidance, sample_cam)
        loss = LOSS_WEIGHT * loss
        dr.backward(loss)
        opt.step()

        if it == 0 or (it + 1) % 10 == 0:
            sensor_fix: mi.Sensor = mi.load_dict({
                'type': 'perspective',
                # 'to_world': mi.scalar_rgb.Transform4f.translate(
                #     box.size * np.asarray(rel_offset),
                # ) @ canon_sensor.world_transform(),
                'to_world': mi.scalar_rgb.Transform4f.look_at(
                    # origin=np.array(canon_transform.translation()) + box.size * np.asarray([0, .4, -2]),
                    # origin=np.array([target_box.center[0], target_box.max[1], target_box.max[2]]),
                    origin=target_box.max,
                    # target=np.array(canon_transform.matrix)[:3, :3] @ np.array([0, 0, 1]) + np.array(canon_transform.translation()),
                    target=center,
                    up=[0, 1, 0]
                ),
                'fov': 60,
                'film': {
                    'type': 'hdrfilm',
                    'width': 512,
                    'height': 512,
                }
            })

            image_fixview = mi.render(scene, sensor=sensor_fix, spp=SPP, seed=it)
            image = mi.util.convert_to_bitmap(image_fixview)
            image = Image.fromarray(np.asarray(image))
            image.save(save_dir / f'fixview_iter_{it:05d}.png')

            guidance_eval_utils = model_guidance.guidance_eval_utils
            guidance_eval_out = model_guidance.guidance_eval(**guidance_eval_utils)
            model_guidance.cleanup_guidance_eval_utils()

            img_save = {'img_render': image_before}
            img_size = image_before.shape[0]
            noise_level = guidance_eval_out['noise_levels'][0]
            for k,v in guidance_eval_out.items():
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    image_item = torch.nn.functional.interpolate(v.permute(0, 3, 1, 2), (img_size, img_size), mode="bilinear").permute(0, 2, 3, 1)
                    img_save.update({
                        k: 255 * image_item[0].detach().cpu().numpy()
                    })
            
            all_imgs = [x for x in img_save.values()]
            all_imgs = np.concatenate(all_imgs, axis=1).astype(np.uint8)
            all_imgs = Image.fromarray(all_imgs)
            all_imgs.save(save_dir / f'{it:05d}_render_noisy_1step_1orig_final_{noise_level:0.4f}.png')

    return scene, tmp_xml_file


def layout_optimize(shape, prompt, save_dir, preset_xml_path, target_box, optimize_iter=1e2):

    prompt = prompt.replace('_', ' ')
    model_prompt_processor = StableDiffusionPromptProcessor({'prompt': prompt})
    model_guidance = StableDiffusionGuidance({
        'pretrained_model_name_or_path': "stabilityai/stable-diffusion-2-1-base",
        'guidance_scale': 100.,
        'min_step_percent': 0.02,
        'max_step_percent': 0.98,
    })


    optimize_iter = int(optimize_iter)
    num_shapes = len(shape)
    shape = _preprocess_shape(shape)
    scene_dict = {'type': 'scene', **{f'{i:02d}': s for i, s in enumerate(shape)}}

    # TODO: remove background things in this scene
    tmp_xml_file = Path(preset_xml_path).with_name(f'tmp_{uuid.uuid4()}.xml')
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        mi.xml.dict_to_xml(scene_dict, tmp_xml_file)
    tree = concatenate_xml_files(preset_xml_path, tmp_xml_file.as_posix())
    tree.write(tmp_xml_file.as_posix())
    scene: mi.Scene = mi.load_file(tmp_xml_file.as_posix())

    ROTATION = 'axis-aligned'
    SDS_WEIGHT = 1e-3 # 0.1
    LR = 1e-4

    if ROTATION == "quaternion":
        # quaternion
        rot_params = nn.Parameter(torch.zeros(num_shapes, 4, device=device), requires_grad=True)
    
    elif ROTATION == 'axis-aligned':
        # axis-rotation
        rot_x_params = nn.Parameter(torch.zeros(num_shapes, 1, device=device), requires_grad=True)
        rot_y_params = nn.Parameter(torch.zeros(num_shapes, 1, device=device), requires_grad=True)
        rot_z_params = nn.Parameter(torch.zeros(num_shapes, 1, device=device), requires_grad=True)
    # axis-translation
    trans_x_params = nn.Parameter(torch.ones(num_shapes, 1, device=device) * 0.01, requires_grad=True)
    trans_y_params = nn.Parameter(torch.zeros(num_shapes, 1, device=device), requires_grad=True)
    trans_z_params = nn.Parameter(torch.zeros(num_shapes, 1, device=device), requires_grad=True)
    scale_params = nn.Parameter(torch.ones(num_shapes, 1, device=device), requires_grad=True)

    # try_setup_sd(prompt)
    optimize_list = [
        {'name': 'trans_x', 'params': list([trans_x_params]), 'lr': LR},
        {'name': 'trans_y', 'params': list([trans_y_params]), 'lr': LR},
        {'name': 'trans_z', 'params': list([trans_z_params]), 'lr': LR},
        {'name': 'rot_x', 'params': list([rot_x_params]), 'lr': LR * 0.},
        {'name': 'rot_y', 'params': list([rot_y_params]), 'lr': LR},
        {'name': 'rot_z', 'params': list([rot_z_params]), 'lr': LR * 0.},
    ]
    
    optimizer = torch.optim.Adam(optimize_list, betas=(0.9, 0.99), eps=1e-15)

    for i in range(optimize_iter):
        bbox = _compute_bbox(scene_dict)
        center = bbox.center
        radius = max(bbox.sizes / 2)

        optimizer.zero_grad()

        # sample elevation, azimuth and camera distance for SDS
        # I use the radius as 1.0 in camera distance unit
        
        sample_cam = random_sample_camera()
        distance = sample_cam['camera_distances'] * torch.Tensor([radius]) * 2
        camera_positions = torch.stack(
            # different from threestudio, in mitsuba3 the y-axis facing upwards
            [
                distance * torch.cos(sample_cam['elevation']) * torch.cos(sample_cam['azimuth']),
                distance * torch.sin(sample_cam['elevation']),
                distance * torch.cos(sample_cam['elevation']) * torch.sin(sample_cam['azimuth']),
            ],
            dim=-1,
        ).reshape(-1).cpu().numpy()
        camera_positions = camera_positions + center

        for k,v in sample_cam.items():
            sample_cam[k] = v.to(device)

        sensor: mi.Sensor = mi.load_dict({
            'type': 'perspective',
            # 'to_world': mi.scalar_rgb.Transform4f.translate(
            #     box.size * np.asarray(rel_offset),
            # ) @ canon_sensor.world_transform(),
            'to_world': mi.scalar_rgb.Transform4f.look_at(
                # origin=np.array(canon_transform.translation()) + box.size * np.asarray([0, .4, -2]),
                # origin=np.array([target_box.center[0], target_box.max[1], target_box.max[2]]),
                origin=camera_positions,
                # target=np.array(canon_transform.matrix)[:3, :3] @ np.array([0, 0, 1]) + np.array(canon_transform.translation()),
                target=center,
                up=[0, 1, 0]
            ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': 64, #256,
                'height': 64, #256,
            }
        })
        render_cam = sensor
        
        if ROTATION == "quaternion":
            rot_init = torch.Tensor([0.1, 0.0, 0.0, 0.0]).reshape(-1, 4).repeat(num_shapes, 1).to(device)
            rot_quaternion = rot_params + rot_init
            rot_quaternion = F.normalize(rot_quaternion, p=2, dim=-1)
            rot_quaternion = rot_quaternion * rot_quaternion.sign()
            # [N, 3, 3]
            rot_pred = quaternion_to_matrix(rot_quaternion)
            # [N, 3, 1]
            trans_pred = torch.cat([trans_x_params, trans_y_params, trans_z_params], dim=-1).unsqueeze(-1)
            # [N, 4, 4]
            mtx = torch.cat([
                torch.cat([
                    rot_pred, trans_pred
                ], dim=-1), torch.Tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(num_shapes, 1, 1).to(device)
            ], dim=1)

            render_img, scene = render_scene_w_pose(scene, cam=render_cam, rotation_rep=ROTATION, mtx=mtx, spp=SPP, seed=i)
        
        elif ROTATION == 'axis-aligned':
            rot_x = torch.sigmoid(rot_x_params) * (0.5 - (-0.5)) + (-0.5)
            rot_y = torch.sigmoid(rot_y_params) * (0.5 - (-0.5)) + (-0.5)
            rot_z = torch.sigmoid(rot_z_params) * (0.5 - (-0.5)) + (-0.5)
            
            # [N, 3]
            trans_pred = torch.cat([trans_x_params, trans_y_params, trans_z_params], dim=-1)
            render_img, scene = render_scene_w_pose(scene, cam=render_cam, rotation_rep=ROTATION, 
                                             rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, trans=trans_pred,
                                             spp=SPP, seed=i)
        from pdb import set_trace; set_trace()
        if i % 10 == 0:
            image = mi.util.convert_to_bitmap(render_img)
            image = Image.fromarray(np.asarray(image))
            image.save(save_dir / f'guidance_iter_{i:05d}.png')

        #     sensor_fix: mi.Sensor = mi.load_dict({
        #         'type': 'perspective',
        #         # 'to_world': mi.scalar_rgb.Transform4f.translate(
        #         #     box.size * np.asarray(rel_offset),
        #         # ) @ canon_sensor.world_transform(),
        #         'to_world': mi.scalar_rgb.Transform4f.look_at(
        #             # origin=np.array(canon_transform.translation()) + box.size * np.asarray([0, .4, -2]),
        #             # origin=np.array([target_box.center[0], target_box.max[1], target_box.max[2]]),
        #             origin=target_box.max,
        #             # target=np.array(canon_transform.matrix)[:3, :3] @ np.array([0, 0, 1]) + np.array(canon_transform.translation()),
        #             target=center,
        #             up=[0, 1, 0]
        #         ),
        #         'fov': 60,
        #         'film': {
        #             'type': 'hdrfilm',
        #             'width': 512,
        #             'height': 512,
        #         }
        #     })

        #     image_fixview = mi.render(scene, sensor=sensor_fix, spp=SPP, seed=i, seed_grad=i+1)
        #     image = mi.util.convert_to_bitmap(image_fixview)
        #     image = Image.fromarray(np.asarray(image))
        #     image.save(save_dir / f'fixview_iter_{i:05d}.png')

        # prompt_utils = model_prompt_processor.__call__()
        # guidance_out = model_guidance.__call__(
        #     render_img.unsqueeze(0), prompt_utils,
        #     elevation=sample_cam['elevation_deg'],
        #     azimuth=sample_cam['azimuth_deg'],
        #     camera_distances=sample_cam['camera_distances'],
        #     rgb_as_latents=False
        # )
        
        # loss = SDS_WEIGHT * guidance_out['loss_sds']

        loss = render_img.mean()
        loss.backward()
        
        # for param in [
        #     trans_x_params,
        #     trans_y_params,
        #     trans_z_params,
        #     rot_x_params,
        #     rot_y_params,
        #     rot_z_params
        # ]:
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             param.grad.data.zero_()

        from pdb import set_trace; set_trace()
        if i % 1 == 0:
            print(f'Training iteration {i+1}/{optimize_iter}, loss: {loss.item()}')

        optimizer.step()

    del model_prompt_processor
    del model_guidance
    return scene, tmp_xml_file


def random_sample_camera(
        batch_size=1,
        elevation_range=[10, 50],
        azimuth_range=[-10, 150],
        camera_distance_range=[1.5, 2.0]
    ):

    if random.random() < 0.5:
        # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
        elevation_deg = (
            torch.rand(batch_size)
            * (elevation_range[1] - elevation_range[0])
            + elevation_range[0]
        )
        elevation = elevation_deg * math.pi / 180
    else:
        # otherwise sample uniformly on sphere
        elevation_range_percent = [
            elevation_range[0] / 180.0 * math.pi,
            elevation_range[1] / 180.0 * math.pi,
        ]
        # inverse transform sampling
        elevation = torch.asin(
            (
                torch.rand(batch_size)
                * (
                    math.sin(elevation_range_percent[1])
                    - math.sin(elevation_range_percent[0])
                )
                + math.sin(elevation_range_percent[0])
            )
        )
        elevation_deg = elevation / math.pi * 180.0
    
    
    # simple random sampling
    azimuth_deg = (
        torch.rand(batch_size)
        * (azimuth_range[1] - azimuth_range[0])
        + azimuth_range[0]
    )
    azimuth = azimuth_deg * math.pi / 180

    # sample distances from a uniform distribution bounded by distance_range
    camera_distances = (
        torch.rand(batch_size)
        * (camera_distance_range[1] - camera_distance_range[0])
        + camera_distance_range[0]
    )

    return {
        'elevation_deg': elevation_deg,
        'elevation': elevation,
        'azimuth_deg': azimuth_deg,
        'azimuth': azimuth,
        'camera_distances': camera_distances
    }
