import json

import torch
import gc
from pathlib import Path
from typing import Optional
import diffusers
from diffusers import DDIMScheduler
import torch.nn.functional as F
from jaxtyping import Float, Bool
from engine.utils.visualize_utils import visualize_depth_map
from PIL import Image
import cv2
import imageio
import numpy as np
import importlib
import numpy.typing
import torchvision
import torchvision.transforms.functional
import torchvision.utils


creg = None
sreg = None
COUNT = None
sizereg = 1
reg_sizes = None
creg_maps = None
sreg_maps = None
text_cond = None
cache_prompts: list[str] = None
cache_cond_embeddings: Float[torch.Tensor, "K 77 768"] = None  # K is the length of `cache_prompts`
cache_cond_embeddings_mod: Float[torch.Tensor, "K 77 768"] = None
cache_neg_prompt: str = None
cache_uncond_embeddings: Float[torch.Tensor, "1 77 768"] = None
cache_segment_start_len: list[tuple[int, int]] = None
cache_prompt2embeddings: dict[str, Float[torch.Tensor, "1 77 768"]] = {}  # currently only support negative prompts

pipe: diffusers.StableDiffusionPipeline = None

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
ATTN_COUNT = 32
SP_SZ = 64
IM_SZ = 512
TK_MAX_LEN = 77


def try_setup_model():
    global pipe

    if pipe is not None:
        return

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        requires_safety_checker=False,
        feature_extractor=None,
        safety_checker=None,
    ).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # FIXME this will modify VAE too..
    mod_count = 0
    for _module in pipe.unet.modules():
        if _module.__class__.__name__ == "Attention":
            _module.__class__.__call__ = mod_forward
            mod_count += 1
    assert mod_count == ATTN_COUNT, mod_count


def cleanup():
    global pipe
    print('[INFO] cleaning up densediffusion...')

    if pipe is None:
        print(f'[INFO] dense diffusion is not used, skipped cleaning up')
        return
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    try:
        importlib.reload(diffusers)  # to undo __class__ modifications
    except ImportError:
        pass

    # TODO what if we don't clean up and use the same dense control for LMD?


def run_sds(
        prompt: str,
        bg_prompt: str,
        docstrings: list[str],
        segm_maps: list[np.typing.NDArray[np.bool_]],
        depth_maps: list[np.typing.NDArray[np.float32]],
        save_dir: str, overwrite: bool,
        save_prefix: Optional[str] = None,
        seed: int = 0, creg_: float = 1, sreg_: float = .3,
        clear_model=False, collapse_prompts=False
):
    raise NotImplementedError()


def run_pipe(prompt: str,
             bg_prompt: str,
             docstrings: list[str],
             segm_maps: list[np.typing.NDArray[np.bool_]],
             depth_maps: list[np.typing.NDArray[np.float32]],
             save_dir: str, overwrite: bool,
             save_prefix: Optional[str] = None,
             seed: int = 0, creg_: float = 1, sreg_: float = .3,
             clear_model=False, collapse_prompts=False
             ):
    save_dir = Path(save_dir)
    save_prefix = '' if save_prefix is None else save_prefix + '_'
    save_path = save_dir / f'{save_prefix}feedforward.png'
    if not overwrite and save_path.exists():
        return
    try_setup_model()
    bsz = 1
    success = process_program(prompt=prompt, bg_prompt=bg_prompt, segm_maps=segm_maps, depth_maps=depth_maps, docstrings=docstrings, bsz=bsz,
                              save_dir=save_dir.as_posix(), save_prefix=save_prefix, clear_model=clear_model,
                              suppress_error=True, collapse_prompts=collapse_prompts)
    if not success:
        return
    global COUNT, sreg, creg
    COUNT = 0
    sreg = sreg_
    creg = creg_

    pipe(None, #[prompt] * bsz,
         generator=torch.Generator().manual_seed(seed),
         prompt_embeds=cache_cond_embeddings[:1].repeat(bsz, 1, 1),
         negative_prompt_embeds=cache_uncond_embeddings.repeat(bsz, 1, 1),
     ).images[0].save(save_path.as_posix())  # will set scheduler timesteps to default 50

    with open((save_dir / f'{save_prefix}_info.json').as_posix(), 'w') as f:
        json.dump({
            'prompt': prompt, 'bg_prompt': bg_prompt, 'docstrings': docstrings,
            'creg': creg, 'sreg': sreg,
        }, f)

    # reset
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    COUNT = None
    sreg = None
    creg = None


def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
    residual = hidden_states

    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)

    global COUNT

    sa_ = True if encoder_hidden_states is None else False
    encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states

    if self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    assert pipe.scheduler.config.num_train_timesteps == 1000, pipe.scheduler.config.num_train_timesteps
    num_inference_steps = pipe.scheduler.num_inference_steps
    if num_inference_steps is None:
        num_inference_steps = pipe.scheduler.config.num_train_timesteps
    if COUNT / ATTN_COUNT < num_inference_steps * 0.3:

        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1],
                                        dtype=query.dtype, device=query.device),
                            query, key.transpose(-1, -2), beta=0, alpha=self.scale)

        treg = torch.pow(pipe.scheduler.timesteps[COUNT // ATTN_COUNT] / pipe.scheduler.config.num_train_timesteps, 5)

        ## reg at self-attn
        if sa_:
            min_value = sim[int(sim.size(0) / 2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0) / 2):].max(-1)[0].unsqueeze(-1)
            mask = sreg_maps[sim.size(1)].repeat(self.heads, 1, 1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads, 1, 1)

            sim[int(sim.size(0) / 2):] += (mask > 0) * size_reg * sreg * treg * (max_value - sim[int(sim.size(0) / 2):])
            sim[int(sim.size(0) / 2):] -= ~(mask > 0) * size_reg * sreg * treg * (
                        sim[int(sim.size(0) / 2):] - min_value)

        ## reg at cross-attn
        else:
            min_value = sim[int(sim.size(0) / 2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0) / 2):].max(-1)[0].unsqueeze(-1)
            mask = creg_maps[sim.size(1)].repeat(self.heads, 1, 1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads, 1, 1)

            sim[int(sim.size(0) / 2):] += (mask > 0) * size_reg * creg * treg * (max_value - sim[int(sim.size(0) / 2):])
            sim[int(sim.size(0) / 2):] -= ~(mask > 0) * size_reg * creg * treg * (
                        sim[int(sim.size(0) / 2):] - min_value)

        attention_probs = sim.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_probs = self.get_attention_scores(query, key, attention_mask)

    COUNT += 1

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    return hidden_states


def compose_maps(
        segm_maps: list[Bool[np.ndarray, "h w"]],
        depth_maps: list[Float[np.ndarray, "h w"]],
        kernel: int = 5,
) -> tuple[list[Bool[np.ndarray, "h w"]], Float[np.ndarray, "h w"], Bool[np.ndarray, "h w"]]:
    kernel = np.ones((kernel, kernel), np.uint8)
    erode_segm_maps = np.stack([cv2.erode((segm_maps[i] * 255).astype(np.uint8), kernel) > 127.5
                                for i in range(len(segm_maps))])

    min_depth_indices = np.argmin(np.where(erode_segm_maps, np.asarray(depth_maps), np.inf), axis=0)
    composite_depth_map = np.take_along_axis(np.stack(depth_maps), min_depth_indices[None, :, :], axis=0)[0]
    segm_maps = [np.logical_and(min_depth_indices == i, segm_maps[i]) for i in range(len(segm_maps))]
    composite_segm_map = np.asarray(segm_maps).sum(0).astype(bool)
    return segm_maps, composite_depth_map, composite_segm_map


def process_program(prompt: str,
                    bg_prompt: str,
                    segm_maps: list[Bool[np.ndarray, "h w"]],
                    depth_maps: list[Float[np.ndarray, "h w"]],
                    docstrings: list[str], bsz: int,
                    save_dir: str, save_prefix: str,
                    neg_prompt: str = '',
                    clear_model: bool = True,
                    suppress_error: bool = False,
                    collapse_prompts: bool = False,
                    ) -> bool:
    if pipe.tokenizer(
            prompt, padding="max_length", return_length=True, return_overflowing_tokens=True,
            max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt",
    )['num_truncated_tokens'] > 0:
        print(f'[ERROR] prompt too long and is truncated: {prompt}')
        if not suppress_error:
            raise ValueError(prompt)
        return False

    global COUNT
    COUNT = None
    assert len(segm_maps) == len(docstrings) == len(depth_maps), (len(segm_maps), len(docstrings), len(depth_maps))

    if save_dir is not None:
        # print(f'[INFO] visualization saved to {save_dir}')
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_prefix = '' if save_prefix is None else save_prefix + '_'

    segm_maps, composite_depth_map, composite_segm_map = compose_maps(segm_maps, depth_maps)
    if save_dir is not None:
        # imageio.imwrite((save_dir / f'{save_prefix}depth.exr').as_posix(), composite_depth_map, format='EXR-FI')
        depth_disp = visualize_depth_map(composite_depth_map, composite_segm_map)
        Image.fromarray((depth_disp * 255).astype(np.uint8)).save((save_dir / f'{save_prefix}depth.png').as_posix())
    # min_depths = [np.where(segm, depth, np.inf).min() for segm, depth in zip(segm_maps, depth_maps)]
    # sorted_indices = np.argsort(min_depths)
    # segm_maps = [segm_maps[i] for i in sorted_indices]
    # depth_maps = [depth_maps[i] for i in sorted_indices]
    # docstrings = [docstrings[i] for i in sorted_indices]
    #
    # acc_segm_map = np.zeros_like(segm_maps[0])
    # composed_segm_maps = []
    # for i in range(len(segm_maps)):
    #     composed_segm_map = np.logical_and(np.logical_not(acc_segm_map), segm_maps[i])
    #     acc_segm_map = np.logical_or(acc_segm_map, segm_maps[i])
    #     composed_segm_maps.append(composed_segm_map)

    # segm_maps = composed_segm_maps

    # layouts_ = [(np.stack(segm_maps).sum(0) == 0).astype(np.uint8)]  # background
    # for ind in range(len(segm_maps)):
    #     # kernel = np.ones((8, 8), np.uint8)
    #     # erode_segm = cv2.erode((segm_maps[ind] * 255).astype(np.uint8), kernel) > 127.5
    #     # distance_to_viewer = depth_maps[ind][erode_segm].min()  # FIXME this is probably wrong
    #     layouts_.append(segm_maps[ind].astype(np.uint8))
    #
    # # TODO!!!! handle occlusion; basically sort boxes by min depth and alpha-compose
    # layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
    # layouts = F.interpolate(torch.cat(layouts), (SP_SZ, SP_SZ), mode='nearest')

    layouts_ = np.stack([np.equal(np.stack(segm_maps).sum(0), 0), *segm_maps])
    layouts = F.interpolate(torch.FloatTensor(layouts_).unsqueeze(1), (SP_SZ, SP_SZ), mode='nearest').cuda()

    if collapse_prompts:
        docstring_to_segm = {}
        for docstring, segm in zip(docstrings, segm_maps):
            if docstring not in docstring_to_segm:
                docstring_to_segm[docstring] = segm
            else:
                docstring_to_segm[docstring] = np.logical_or(docstring_to_segm[docstring], segm)
        collapsed_docstrings, collapsed_segm = map(list, zip(*docstring_to_segm.items()))
        collapsed_layouts_ = np.stack([np.equal(np.stack(collapsed_segm).sum(0), 0), *collapsed_segm])
        collapsed_layouts = F.interpolate(torch.FloatTensor(collapsed_layouts_).unsqueeze(1), (SP_SZ, SP_SZ), mode='nearest').cuda()
    else:
        collapsed_docstrings = docstrings
        collapsed_layouts = layouts

    if save_dir is not None:
        torchvision.transforms.functional.to_pil_image(
            torchvision.utils.draw_segmentation_masks(
                image=torch.ones((3, *layouts_[0].shape), dtype=torch.uint8) * 255,
                masks=torch.BoolTensor(layouts_),
            ),
        ).save((save_dir / f'{save_prefix}boxes.png').as_posix())

        if collapse_prompts:
            torchvision.transforms.functional.to_pil_image(
                torchvision.utils.draw_segmentation_masks(
                    image=torch.ones((3, *collapsed_layouts_[0].shape), dtype=torch.uint8) * 255,
                    masks=torch.BoolTensor(collapsed_layouts_),
                ),
            ).save((save_dir / f'{save_prefix}collapsed_boxes.png').as_posix())

    """ encode texts """

    global cache_cond_embeddings, cache_cond_embeddings_mod, cache_prompts, cache_segment_start_len
    prompts = [prompt, bg_prompt, *collapsed_docstrings]
    if prompts != cache_prompts:
        if pipe.tokenizer(
                prompts[0], padding="max_length", return_length=True, return_overflowing_tokens=True,
                max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt",
        )['num_truncated_tokens'] > 0:
            print(f'[ERROR] prompt too long and is truncated: {prompts}')
            if not suppress_error:
                raise ValueError(prompts)
            return False
        text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = pipe.text_encoder(text_input.input_ids.to('cuda'))[0]
        cond_embeddings_mod = cond_embeddings.clone()

        cache_segment_start_len = [(None, None)] * len(prompts)
        used_segments = set()  # each prompt will be mapped to a distinct segment
        for i in range(1, len(prompts)):
            wlen = (text_input['length'][i] - 2).item()
            widx = text_input['input_ids'][i][1:1 + wlen]
            for j in range(TK_MAX_LEN - wlen + 1):
                segment_found = False
                if (text_input['input_ids'][0][j:j + wlen] == widx).sum() == wlen and (j, wlen) not in used_segments:
                    cache_segment_start_len[i] = (j, wlen)
                    cond_embeddings_mod[0][j:j + wlen] = cond_embeddings_mod[i][1:1 + wlen]
                    used_segments.add((j, wlen))
                    segment_found = True
                    break
            if not segment_found:
                raise ValueError(prompts)
                import ipdb;
                ipdb.set_trace()
        # for i in range(len(prompts)):
        #     print(prompts[i], cache_segment_start_len[i])
        cache_cond_embeddings = cond_embeddings
        cache_cond_embeddings_mod = cond_embeddings_mod
        cache_prompts = prompts

    global cache_neg_prompt, cache_uncond_embeddings, cache_prompt2embeddings
    if neg_prompt != cache_neg_prompt:
        if neg_prompt in cache_prompt2embeddings:
            uncond_embeddings = cache_prompt2embeddings[neg_prompt]
        else:
            print(f'[INFO] computing embeddings for negative prompt: {neg_prompt} and adding it to cache')
            uncond_input = pipe.tokenizer([neg_prompt], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to('cuda'))[0]
            cache_prompt2embeddings[neg_prompt] = uncond_embeddings
        cache_neg_prompt = neg_prompt
        cache_uncond_embeddings = uncond_embeddings

    """ self-attention reg """

    global sreg_maps, reg_sizes
    sreg_maps = {}
    reg_sizes = {}
    for r in range(4):
        res = int(SP_SZ / np.power(2, r))
        layouts_s = F.interpolate(layouts, (res, res), mode='nearest')
        layouts_s = (layouts_s.view(layouts_s.size(0), 1, -1) * layouts_s.view(layouts_s.size(0), -1, 1)).sum(
            0).unsqueeze(0).repeat(bsz, 1, 1)
        reg_sizes[np.power(res, 2)] = 1 - sizereg * layouts_s.sum(-1, keepdim=True) / (np.power(res, 2))
        sreg_maps[np.power(res, 2)] = layouts_s

    """ cross-attention reg """

    # FIXME what happens when there are multiple entities with the same prompt?

    global creg_maps
    pww_maps = torch.zeros(1, TK_MAX_LEN, SP_SZ, SP_SZ, device='cuda')
    for i in range(1, len(prompts)):
        j, wlen = cache_segment_start_len[i]
        pww_maps[:, j:j + wlen, :, :] = collapsed_layouts[i - 1:i]

    creg_maps = {}
    for r in range(4):
        res = int(SP_SZ / np.power(2, r))
        layout_c = F.interpolate(pww_maps, (res, res), mode='nearest').view(1, TK_MAX_LEN, -1).permute(0, 2, 1).repeat(bsz, 1, 1)
        creg_maps[np.power(res, 2)] = layout_c

    """ text_cond """

    global text_cond
    text_cond = torch.cat([cache_uncond_embeddings.repeat(bsz, 1, 1), cache_cond_embeddings_mod[:1].repeat(bsz, 1, 1)])
    if clear_model and pipe.tokenizer is not None:
        print(f'[INFO] clearing tokenizer and text encoder')
        pipe.tokenizer = None
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
    return True
