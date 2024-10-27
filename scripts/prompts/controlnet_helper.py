try:
    import cv2
except Exception:
    cv2 = None
from functools import partial
import requests
from typing import Callable, Literal
import torch
import os
import gc
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# https://huggingface.co/spaces/hysts/ControlNet-v1-1

__all__ = ['run_controlnet_canny', 'run_controlnet_softedge',
           'run_controlnet_seg', 'run_controlnet_mlsd',
           'clear_controlnet_model',
           ]


root = Path(__file__).parent.parent


initialized = False
model: str = None
processor: Callable[[Image.Image], Image.Image] = None
pipe: StableDiffusionControlNetPipeline = None


def try_setup_model_softedge():
    from controlnet_aux import PidiNetDetector

    global initialized, processor, pipe, model

    if initialized:
        assert model == 'softedge', model
        return
    initialized = True
    model = 'softedge'

    checkpoint = "lllyasviel/control_v11p_sd15_softedge"
    # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()


def try_setup_model_canny():
    global initialized, processor, pipe, model

    if initialized:
        assert model == 'canny', model
        return
    initialized = True
    model = 'canny'

    checkpoint = "lllyasviel/control_v11p_sd15_canny"

    low_threshold = 100
    high_threshold = 200

    def processor(image: Image.Image) -> Image.Image:
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()


def try_setup_model_seg():
    global initialized, processor, pipe, model

    if initialized:
        assert model == 'seg', model
        return
    initialized = True
    model = 'seg'

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    checkpoint = "lllyasviel/control_v11p_sd15_seg"

    def processor(image: Image.Image) -> Image.Image:
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        for label, color in enumerate(ada_palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        return Image.fromarray(color_seg)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()


def try_setup_model_mlsd():
    from controlnet_aux import MLSDdetector
    global initialized, processor, pipe, model

    if initialized:
        assert model == 'mlsd', model
        return
    initialized = True
    model = 'mlsd'

    checkpoint = "lllyasviel/control_v11p_sd15_mlsd"

    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()


TRY_SETUPS = {
    'canny': try_setup_model_canny,
    'softedge': try_setup_model_softedge,
    'seg': try_setup_model_seg,
    'mlsd': try_setup_model_mlsd,
}


def run_pipe(image: str,
             prompt: str, save_dir: str,
             model: Literal['canny', 'softedge', 'seg', 'mlsd'],
             save_prefix: str | None = None,
             overwrite: bool = True,
):
    if not torch.cuda.is_available():
        return

    save_prefix = '' if save_prefix is None else f'{save_prefix}_'

    save_dir = Path(save_dir)
    control_save_path = save_dir / f"{save_prefix}{model}_control.png"
    image_save_path = save_dir / f"{save_prefix}{model}_output.png"
    if control_save_path.exists() and image_save_path.exists() and not overwrite:
        return

    TRY_SETUPS[model]()

    image = Image.open(image)
    control_image = processor(image)
    control_image.save(control_save_path.as_posix())

    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(image_save_path.as_posix())


def clear_controlnet_model():
    global initialized, model, processor, pipe
    initialized = False
    model = ''
    del processor
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    processor = None
    pipe = None


run_controlnet_canny = partial(run_pipe, model='canny')
run_controlnet_softedge = partial(run_pipe, model='softedge')
run_controlnet_seg = partial(run_pipe, model='seg')
run_controlnet_mlsd = partial(run_pipe, model='mlsd')


def main():
    save_dir = root / 'outputs/run_controlnet'
    save_dir.mkdir(exist_ok=True, parents=True)

    image = "https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/main/images/input.png"
    image = requests.get(image, stream=True).raw
    prompt = "royal chamber with fancy bed"
    run_pipe(image, prompt, save_dir.as_posix(), 'canny')


ada_palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])


if __name__ == "__main__":
    main()
