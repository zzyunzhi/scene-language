import torch
import pickle
import base64
from pathlib import Path
from engine.constants import PROJ_DIR
import numpy as np
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

MAX_COLORS = 12


def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix


def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask


with open('/viscam/projects/concepts/third_party/DenseDiffusion/dataset/valset.pkl', 'rb') as f:
    val_prompt = pickle.load(f)
val_layout = '/viscam/projects/concepts/third_party/DenseDiffusion/dataset/valset_layout/'


examples = [[val_layout + '0.png',
             '***'.join([val_prompt[0]['textual_condition']] + val_prompt[0]['segment_descriptions']), 381940206],
            [val_layout + '1.png',
             '***'.join([val_prompt[1]['textual_condition']] + val_prompt[1]['segment_descriptions']), 307504592],
            [val_layout + '5.png',
             '***'.join([val_prompt[5]['textual_condition']] + val_prompt[5]['segment_descriptions']),
             114972190]]


def process_example(layout=examples[1][0], all_prompts=examples[1][1], save_dir=None, save_prefix=None):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_prefix = '' if save_prefix is None else save_prefix + '_'
    all_prompts = all_prompts.split('***')
    general_prompt = all_prompts[0]
    prompts = all_prompts[1:]

    binary_matrixes = []

    if isinstance(layout, str):
        layout = Image.open(layout)
    im2arr = np.array(layout)[:, :, :3]
    unique, counts = np.unique(np.reshape(im2arr, (-1, 3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)

    binary_matrix = create_binary_matrix(im2arr, (0, 0, 0))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_ * (255, 255, 255) + (1 - binary_matrix_) * (50, 50, 50)
    if save_dir is not None:
        Image.fromarray(colored_map.astype(np.uint8)).save(save_dir / f'{save_prefix}{general_prompt.replace(" ", "_")}.png')

    for i in range(len(all_prompts) - 1):
        r, g, b = unique[sorted_idx[i]]
        if any(c != 255 for c in (r, g, b)) and any(c != 0 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r, g, b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
            if save_dir is not None:
                Image.fromarray(colored_map.astype(np.uint8)).save(save_dir / f'{save_prefix}{i:02d}_{prompts[i].replace(" ", "_")}.png')
    # for ind in range(1, len(binary_matrixes)):
    #     if prompts[ind] == 'a yellowish full moon':
    #         # return [binary_matrixes[0], binary_matrixes[ind]], 'a yellowish full moon in the night sky', [prompts[0], prompts[ind]]
    #         # return [binary_matrixes[ind]], 'a yellowish full moon in the night sky', [prompts[ind]]
    #         return [binary_matrixes[0], binary_matrixes[ind]], 'a yellowish full moon in sky', ['sky', 'a yellowish full moon']
    #         return [binary_matrixes[0], binary_matrixes[ind]], 'a yellowish full moon', ['', 'a yellowish full moon']
    return binary_matrixes, general_prompt, prompts
