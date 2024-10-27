import shutil
import sys
import cv2
import torch
import os
import numpy as np
import numpy.typing
from typing import Union, Optional
from PIL import Image
from engine.constants import PROJ_DIR, DEBUG
from pathlib import Path
import gc

root = Path(__file__).parent.parent

sys.path.append((Path(PROJ_DIR) / "engine/third_party/loose_control").resolve().as_posix())

from engine.third_party.loose_control.loosecontrol import LooseControlNet

model: Union[LooseControlNet, None] = None


def try_setup_model():
    global model
    if model is not None:
        return
    model = LooseControlNet()
    model.to(torch_device='cuda')


def clearup_model():
    global model
    if model is None:
        return
    model = None
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def run_pipe(depth: np.typing.NDArray[np.float32], segm: np.typing.NDArray[np.bool_], prompt: str, save_dir: str, overwrite: bool, save_prefix: Optional[str] = None):
    # https://github.com/shariqfarooq123/LooseControl/blob/main/app.py
    save_dir = Path(save_dir)
    # if save_prefix is not None and (save_dir / f'{save_prefix}image.png').exists():  # FIXME legacy hack
    #     shutil.copy((save_dir / f'{save_prefix}image.png').as_posix(), (save_dir / f'{save_prefix}_image.png').as_posix())

    save_prefix = f'{save_prefix}_' if save_prefix is not None else ''
    save_path = save_dir / f'{save_prefix}image.png'
    if not overwrite and save_dir.exists() and save_path.exists():
        print(f'skipping {save_dir=}, {os.listdir(save_dir)}')
        return
    save_dir.mkdir(exist_ok=True, parents=True)

    if segm.any():
        kernel = np.ones((8, 8), np.uint8)
        erode_segm = cv2.erode((segm * 255).astype(np.uint8), kernel) > 127.5
        valid_disp = 1 / depth[erode_segm]
        dmax = np.quantile(valid_disp, 0.95)
        image_input = np.zeros_like(depth)
        image_input[erode_segm] = (valid_disp / dmax).clip(min=0, max=1)  # FIXME not sure
    else:
        image_input = np.zeros_like(depth)

    image_input = Image.fromarray((image_input * 255).astype(np.uint8)).convert("RGB").resize((512, 512), resample=Image.BILINEAR)
    image_input.save(save_dir / f'{save_prefix}depth.png')
    if DEBUG:
        return

    try_setup_model()
    negative_prompt = "blurry, text, caption, lowquality, lowresolution, low res, grainy, ugly"

    image = model(prompt=prompt.capitalize(), negative_prompt=negative_prompt, control_image=image_input,
                  controlnet_conditioning_scale=1.0, generator=torch.Generator().manual_seed(0),
                  num_inference_steps=20)

    image.save(save_path.as_posix())
