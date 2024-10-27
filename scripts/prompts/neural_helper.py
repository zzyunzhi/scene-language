import torch
from pathlib import Path
import os
from diffusers.utils import export_to_gif
from tu.train_setup import set_seed_benchmark
from engine.utils.mesh_utils import rewrite_mesh
from engine.constants import PROJ_DIR

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

from PIL import Image
from diffusers import DiffusionPipeline
import numpy as np
if False:
    import rembg
    import trimesh
    import sys
    sys.path.append(Path('/viscam/projects/concepts/zzli/engine', 'engine/third_party/TripoSR').resolve().as_posix())
    # sys.path.append(Path(PROJ_DIR, 'engine/third_party/TripoSR').resolve().as_posix())
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import gc

xm = None
model = None
diffusion = None
device = torch.device('cuda')

tsr_model = None
sd_pipeline = None

# shape_e_save_subdir = Path(PROJ_DIR) / 'cache/shap_e'
shape_e_save_subdir = Path('/viscam/projects/concepts/zzli/engine/cache/shap_e')


def get_cache_save_dir(prompt: str) -> str:
    if prompt == '':
        prompt_save_subdir = shape_e_save_subdir / '_'
    else:
        prompt_save_subdir = shape_e_save_subdir / prompt.replace(' ', '_')
    return prompt_save_subdir.as_posix()


def try_setup_model():
    print('[INFO] setting up model for neural_helper.py...')
    global xm, model, diffusion
    if xm is not None:
        return
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))


def clearup_model():
    global xm, model, diffusion
    if xm is None:
        return
    xm, model, diffusion = None, None, None
    gc.collect()
    torch.cuda.empty_cache()


def try_setup_tsr_model():
    global tsr_model
    if tsr_model is not None:
        return
    tsr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    tsr_model.renderer.set_chunk_size(8192)
    tsr_model.to(device)


def try_setup_sd_model():
    global sd_pipeline
    sd_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    sd_pipeline.to(device)


def run_TripoSR(prompt: str, save_dir: str, overwrite: bool):
    save_dir = Path(save_dir)

    if not overwrite and save_dir.exists() and len(list(save_dir.rglob('*.ply'))) > 0:
        # print(f'skipping {save_dir=}, {os.listdir(save_dir)}')
        return
    
    try_setup_tsr_model()
    try_setup_sd_model()
    
    print('running', prompt, save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mc_resolution = 256
    foreground_ratio = 0.9
    i = 0

    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    # TODO: now just use one example image, 
    # need to further implement text-to-image
    sd_prompt = prompt + ', foreground object, realistic, sharp'
    with torch.no_grad():
        prompt_image = sd_pipeline(sd_prompt).images[0]
    prompt_image_save_path = (save_dir / f't2i_{i:02d}.png').as_posix()
    prompt_image.save(prompt_image_save_path)
    
    rembg_session = rembg.new_session()
    gen_image = Image.open(prompt_image_save_path).convert('RGB')
    gen_image = remove_background(gen_image, rembg_session)
    gen_image = resize_foreground(gen_image, foreground_ratio)
    image = fill_background(gen_image)
    seg_image_save_path = (save_dir / f'seg_{i:02d}.png').as_posix()
    image.save(seg_image_save_path)
    
    scene_codes = tsr_model(image, device=device)
    mesh = tsr_model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    # mesh = to_gradio_3d_orientation(mesh)
    
    ply_save_path = (save_dir / f'{i:02d}.ply').as_posix()
    # origin_ply_save_path = (save_dir / f'{i:02d}_origin_color.ply').as_posix()
    # mesh.export(origin_ply_save_path)
    # rescale mesh vertex color to [0, 1] for further process
    # scaled_vertex_colors = mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255
    # mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=scaled_vertex_colors)
    mesh.export(ply_save_path)
    rewrite_mesh(ply_save_path)


def run_pipe(prompt: str, save_dir: str, overwrite: bool):

    save_dir = Path(save_dir)

    if not overwrite and save_dir.exists() and len(list(save_dir.rglob('*.ply'))) > 0:
        # print(f'skipping {save_dir=}, {os.listdir(save_dir)}')
        return
    print('running', prompt, save_dir)
    try_setup_model()
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1 #4
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=False,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    render_mode = 'nerf' # you can change this to 'stf'
    # size = 64 # this is the size of the renders; higher values take longer to render.
    size = 128

    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        gif_save_path = (save_dir / f'{i:02d}.gif').as_posix()
        _ = export_to_gif(images, gif_save_path)

    for i, latent in enumerate(latents):
        ply_save_path = (save_dir / f'{i:02d}.ply').as_posix()

        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(ply_save_path, 'wb') as f:
            t.write_ply(f)

        rewrite_mesh(ply_save_path)

        # obj_save_path = (Path(save_dir) / f'{i:02d}.obj').as_posix()
        # with open(obj_save_path, 'w') as f:
        #     t.write_obj(f)
        # rewrite_mesh(obj_save_path)


if __name__ == "__main__":
    # http://vcv.stanford.edu/cgi-bin/file-explorer/?dir=%2Fviscam%2Fprojects%2Fconcepts%2Fengine%2Fscripts%2Fexp%2Ficl_0417%2Foutputs%2Fneural_helper&patterns_show=*&patterns_highlight=&w=375&h=375&n=4&autoplay=1&showmedia=1
    set_seed_benchmark(0)
    for prompt in ['a cage', 'a green chair', 'a single domino piece','a domino']:
        run_pipe(prompt=prompt,
                 save_dir=(Path(__file__).parent.parent / 'outputs/neural_helper' / prompt.replace(' ', '_')).as_posix(),
                 overwrite=True)
