import shutil
from typing import Union
from PIL import Image
import sys
import os
from pathlib import Path
from engine.constants import PROJ_DIR, DEBUG
from engine.utils.omost_utils import MyCanvas
import gc
import torch

model = None


def try_setup_model():
    global model
    if model is not None:
        return

    sys.path.insert(0, (Path(PROJ_DIR) / 'engine/third_party/omost').as_posix())
    # import lib_omost.memory_management as memory_management
    # memory_management.high_vram = True
    from engine.third_party.omost.gradio_app import diffusion_fn
    model = diffusion_fn


def clearup_model():
    global model
    if model is None:
        return
    model = None
    gc.collect()
    torch.cuda.empty_cache()


def run_pipe(canvas: MyCanvas, save_dir: str, save_prefix: Union[str, None]):
    try_setup_model()
    canvas_outputs = canvas.dense_process()

    # positive_result = []
    # positive_pooler = None
    #
    # for item in canvas_outputs['bag_of_conditions']:
    #     current_mask = torch.from_numpy(item['mask']).to(torch.float32)
    #     current_prefixes = item['prefixes']
    #     current_suffixes = item['suffixes']
    #
    #     current_cond = pipeline.encode_bag_of_subprompts_greedy(prefixes=current_prefixes, suffixes=current_suffixes)
    #
    #     if positive_pooler is None:
    #         positive_pooler = current_cond['pooler']
    #
    #     positive_result.append((current_mask, current_cond['cond']))
    if DEBUG:
        return

    chatbot = model(
        chatbot=[], canvas_outputs=canvas_outputs,
        num_samples=1, seed=12345, image_width=1024, image_height=1024, highres_scale=1,
        steps=25, cfg=5, highres_steps=20, highres_denoise=.4,
        negative_prompt='lowres, bad anatomy, bad hands, cropped, worst quality',
    )
    save_prefix = '' if save_prefix is None else save_prefix + '_'
    Image.fromarray(canvas_outputs['initial_latent']).save((Path(save_dir) / f'{save_prefix}latents.png').as_posix())
    for ind, item in enumerate(chatbot):
        _, (image_path, _) = item
        shutil.copy(image_path, (Path(save_dir) / f'{save_prefix}image_{ind:02d}.png').as_posix())
