from engine.utils.type_utils import BBox
import torch
import random
import numpy as np
import sys
from pathlib import Path
from engine.constants import PROJ_DIR, DEBUG
from diffusers import EulerDiscreteScheduler
import os

migc_root = Path('/viscam/projects/concepts/third_party/MIGC')
pipe = None


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def try_setup_model():
    global pipe
    if pipe is not None:
        return
    sys.path.insert(0, migc_root.as_posix())
    from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
    from migc.migc_utils import load_migc

    pipe = StableDiffusionMIGCPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.attention_store = AttentionStore()
    load_migc(pipe.unet, pipe.attention_store,
              (migc_root / 'pretrained_weights/MIGC_SD14.ckpt').as_posix(), attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


def run_pipe(boxes: list[tuple[str, BBox]],
             prompt: str,  # root prompt only
             save_dir: str, resolution: int,
             overwrite: bool = False,
             save_prefix: str | None = None):
    save_dir = Path(save_dir)
    save_prefix = '' if save_prefix is None else save_prefix + '_'
    save_path = save_dir / f'{save_prefix}output.png'
    if not overwrite and save_path.exists():
        return
    try_setup_model()

    # https://github.com/limuloo/MIGC/blob/0eca21f07cc8ab0ab1aad6fbdd75846838650b71/migc_gui/app.py#L63
    width = height = resolution
    prompt_final = [[prompt]]
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    bboxes = [[]]
    for i in range(len(boxes)):
        docstring, box = boxes[i]
        prompt_final[0].append(docstring)
        prompt_final[0][0] += ',' + docstring
        l = box.min[0] / width
        u = box.min[1] / height
        r = l + box.sizes[0] / width
        d = u + box.sizes[1] / height
        bboxes[0].append([l, u, r, d])
    seed_everything(0)
    try:
        image = pipe(prompt_final, bboxes, num_inference_steps=50, guidance_scale=7.5,
                     MIGCsteps=25, NaiveFuserSteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
    except Exception:
        import traceback
        traceback.print_exc()
        return
    image.save(save_path.as_posix())
    image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
    image.save(save_path.with_stem(save_path.stem + '_anno'))


if __name__ == '__main__':
    prompt_final = [['masterpiece, best quality,'
                     'black colored ball,gray colored cat,white colored  bed,green colored plant,red colored teddy bear,blue colored wall,brown colored vase,orange colored book,yellow colored hat',
                     'black colored ball', 'gray colored cat', 'white colored  bed',
                     'green colored plant', \
                     'red colored teddy bear', 'blue colored wall', 'brown colored vase', 'orange colored book',
                     'yellow colored hat']]
    bboxes = [[[0.3125, 0.609375, 0.625, 0.875], [0.5625, 0.171875, 0.984375, 0.6875], \
               [0.0, 0.265625, 0.984375, 0.984375], [0.0, 0.015625, 0.21875, 0.328125], \
               [0.171875, 0.109375, 0.546875, 0.515625], [0.234375, 0.0, 1.0, 0.3125], \
               [0.71875, 0.625, 0.953125, 0.921875], [0.0625, 0.484375, 0.359375, 0.8125], \
               [0.609375, 0.09375, 0.90625, 0.28125]]]
    try_setup_model()