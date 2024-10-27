import difflib
import spacy
import json
import itertools


# engine-specific helper has access to engine-agnostic helper
from typing import Optional, List
from pathlib import Path
import time
from type_utils import T, Shape, P
from _shape_utils import primitive_call
from math_utils import _scale_matrix
from minecraft_types import valid_blocks

from shape_utils import *
from math_utils import *


__all__ = ["execute"]


nearest_block_cache = {}


def prepare_dir_for_exec(
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    description: Optional[str] = None,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / f"outputs/helper_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = "" if save_prefix is None else save_prefix + "_"
    description = description if description else f"minecraft_exp_{timestamp}"

    return save_dir, save_prefix, description


def execute_animation(
    frames: List[Shape],
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    save_dir, save_prefix, description = prepare_dir_for_exec(
        save_dir, save_prefix, description
    )
    all_shapes = list(itertools.chain(*frames))

    # 1. Extract scale of entire scene
    # This is global across all frames
    x_pad, y_pad, z_pad, width, height, length = get_x_y_z_boundaries(all_shapes)
    data = {"width": width, "height": height, "depth": length, "frames": []}

    # 2. Place blocks in region
    for frame in frames:
        blocks = []
        for shape in frame:
            if shape["type"] == "block":
                blocks.append(place_cuboid(shape, init_coords=(x_pad, y_pad, z_pad)))
        data["frames"].append(blocks)

    # 3. Save
    save_dir = Path(save_dir)
    save_prefix = "" if save_prefix is None else save_prefix + "_"
    output_path = str(save_dir / f"{save_prefix}_{description}.json")

    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Writing to {output_path}")


def execute(
    shapes: Shape,
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    save_dir, save_prefix, description = prepare_dir_for_exec(
        save_dir, save_prefix, description
    )

    # 1. Extract scale of entire scene
    x_pad, y_pad, z_pad, width, height, length = get_x_y_z_boundaries(shapes)
    data = {"width": width, "height": height, "depth": length, "frames": []}

    # 2. Place blocks in region
    frame = []
    for s in shapes:
        if s["type"] == "block":
            frame.append(place_cuboid(s, init_coords=(x_pad, y_pad, z_pad)))

    data["frames"].append(frame)

    # 3. Save
    save_dir = Path(save_dir)
    save_prefix = "" if save_prefix is None else save_prefix + "_"
    output_path = str(save_dir / f"{save_prefix}_{description}.json")

    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Writing to {output_path}")


def place_cuboid(data, init_coords):
    """
    Place blocks in a scaled and translated area.

    :param data: A dictionary containing the block type and transformation matrix
                    Example format:
                    {
                        'type': 'block',
                        'block_type': 'minecraft:light_blue_concrete',
                        'to_world': [[1.5, 0, 0, 3],
                                    [0, 1.5, 0, 0],
                                    [0, 0, 1.5, 0],
                                    [0, 0, 0, 1]],
                        'fill': True,
                        'info': {
                            "stack": [
                                ("leaves", UUID("d32489d8-8a95-4f2b-8bd0-624bc64d244f")),
                                ("simple_tree", UUID("4361ae4d-45d4-411a-b668-6500e823c202")),
                                ("forest", UUID("166d1207-2a82-40a5-9b99-366d2ef45d60")),
                            ]
                        }
                    }
    """
    # Extract stack
    stack = []
    for s in data["info"]["stack"]:
        stack.append(s[0])

    x0, y0, z0 = init_coords
    scale_x, scale_y, scale_z, translation_x, translation_y, translation_z = (
        extract_scale_translation(data)
    )

    # Adjust relative to region's init coord
    translation_x, translation_y, translation_z = (
        translation_x - x0,
        translation_y - y0,
        translation_z - z0,
    )

    # Protections for bad inputs
    block_type = remap_problematic_mc_blocks(data["block_type"])
    block_type = find_closest_match(block_type, valid_blocks)

    # Grab the default kwargs, this could be empty {}
    default_kwargs = get_default_block_kwargs(block_type)
    # Merge default_kwargs with the given block_kwargs
    if default_kwargs:
        if not data["block_kwargs"]:
            data["block_kwargs"] = default_kwargs
        else:
            for key, value in default_kwargs.items():
                if key not in data["block_kwargs"]:
                    data["block_kwargs"][key] = value

    return {
        "start": [translation_x, translation_y, translation_z],
        "end": [
            translation_x + scale_x,
            translation_y + scale_y,
            translation_z + scale_z,
        ],
        "fill": data["fill"],
        "type": block_type,
        "properties": data["block_kwargs"],
        "stack": stack,
    }


def get_x_y_z_boundaries(shapes):
    x_boundary = [float("inf"), float("-inf")]
    y_boundary = [float("inf"), float("-inf")]
    z_boundary = [float("inf"), float("-inf")]

    for s in shapes:
        scale_x, scale_y, scale_z, translation_x, translation_y, translation_z = (
            extract_scale_translation(s)
        )

        # Compare against x_boundary
        x_0, x_1 = translation_x, translation_x + scale_x - 1
        x_boundary[0] = min(x_boundary[0], x_0)
        x_boundary[1] = max(x_boundary[1], x_1)

        # Compare against y_boundary
        y_0, y_1 = translation_y, translation_y + scale_y - 1
        y_boundary[0] = min(y_boundary[0], y_0)
        y_boundary[1] = max(y_boundary[1], y_1)

        # Compare against z_boundary
        z_0, z_1 = translation_z, translation_z + scale_z - 1
        z_boundary[0] = min(z_boundary[0], z_0)
        z_boundary[1] = max(z_boundary[1], z_1)

    x, width = x_boundary[0], x_boundary[1] - x_boundary[0] + 1
    y, height = y_boundary[0], y_boundary[1] - y_boundary[0] + 1
    z, length = z_boundary[0], z_boundary[1] - z_boundary[0] + 1

    return x, y, z, width, height, length


def extract_scale_translation(data):
    matrix = data["to_world"]

    # Scaling factors (assuming uniform scaling for simplicity)
    scale_x, scale_y, scale_z = matrix[0][0], matrix[1][1], matrix[2][2]
    # Translation components
    translation_x, translation_y, translation_z = (
        matrix[0][3],
        matrix[1][3],
        matrix[2][3],
    )

    return (
        int(scale_x),
        int(scale_y),
        int(scale_z),
        int(translation_x),
        int(translation_y),
        int(translation_z),
    )


cuboid_fn = lambda block_type, block_kwargs={}, scale=(1, 1, 1), fill=True: [
    {
        "type": "block",
        "block_type": block_type,
        "block_kwargs": block_kwargs,
        "fill": fill,
        "to_world": _scale_matrix(scale, enforce_uniform=False),
        "info": {"stack": []},
    }
]


delete_fn = lambda scale=(1, 1, 1): [
    {
        "type": "block",
        "block_type": "minecraft:air",
        "block_kwargs": {},
        "fill": True,
        "to_world": _scale_matrix(scale, enforce_uniform=False),
        "info": {"stack": []},
    }
]


def impl_primitive_call():
    def fn(name, **kwargs):
        return {"set_cuboid": cuboid_fn, "delete_blocks": delete_fn}[name](**kwargs)

    return fn


primitive_call.implement(impl_primitive_call)


##### HELPER TO GET NEAREST SEMANTIC BLOCK
# Load the spaCy model
nlp = spacy.load("en_core_web_md")


# Function to calculate character-level similarity
def character_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


# Function to calculate semantic similarity
def semantic_similarity(query_doc, block_doc):
    return query_doc.similarity(block_doc)


# Function to process minecraft types
def parse_minecraft_types(minecraft_type):
    prefix = "minecraft:"
    prefix_len = len(prefix)
    result = minecraft_type[prefix_len:]

    return result.replace("_", " ")


# Function to find the closest match using both character-level and semantic similarity
def find_closest_match(query, valid_blocks, char_weight=0.6, sem_weight=0.4):
    if query in valid_blocks:
        return query

    if query in nearest_block_cache:
        return nearest_block_cache[query]

    # Remove prefix
    query_no_pre = parse_minecraft_types(query)

    valid_blocks = list(valid_blocks)
    query_doc = nlp(query_no_pre.replace("_", " "))
    max_similarity = -1
    closest_match = None

    for block in valid_blocks:
        # Replace underscores
        block_no_pre = parse_minecraft_types(block)
        block_doc = nlp(block_no_pre)
        char_sim = character_similarity(query, block)
        sem_sim = semantic_similarity(query_doc, block_doc)
        combined_sim = (char_weight * char_sim) + (sem_weight * sem_sim)

        if combined_sim > max_similarity:
            max_similarity = combined_sim
            closest_match = block

    nearest_block_cache[query] = closest_match

    print(f"[WARNING] Replacing {query} with {closest_match}")
    return closest_match


# Function to get the default block_kwargs in case none are provided
def get_default_block_kwargs(block_type):
    defaults = {
        "minecraft:oak_log": {"axis": "y"},
        "minecraft:spruce_log": {"axis": "y"},
        "minecraft:water": {"level": "0"},
        "minecraft:lava": {"level": "0"},
        "minecraft:oak_stairs": {
            "facing": "north",
            "half": "bottom",
            "shape": "straight",
        },
        "minecraft:stone_stairs": {
            "facing": "north",
            "half": "bottom",
            "shape": "straight",
        },
        "minecraft:oak_slab": {"type": "bottom"},
        "minecraft:stone_slab": {"type": "bottom"},
        "minecraft:oak_door": {
            "facing": "north",
            "half": "lower",
            "hinge": "left",
            "open": "false",
            "powered": "false",
        },
        "minecraft:iron_door": {
            "facing": "north",
            "half": "lower",
            "hinge": "left",
            "open": "false",
            "powered": "false",
        },
        "minecraft:white_bed": {"facing": "north", "part": "foot", "occupied": "false"},
        "minecraft:red_bed": {"facing": "north", "part": "foot", "occupied": "false"},
        "minecraft:oak_trapdoor": {
            "facing": "north",
            "half": "bottom",
            "open": "false",
            "powered": "false",
        },
        "minecraft:iron_trapdoor": {
            "facing": "north",
            "half": "bottom",
            "open": "false",
            "powered": "false",
        },
        "minecraft:piston": {"facing": "north", "extended": "false"},
        "minecraft:sticky_piston": {"facing": "north", "extended": "false"},
        "minecraft:redstone_wire": {
            "north": "none",
            "south": "none",
            "west": "none",
            "east": "none",
        },
        "minecraft:oak_leaves": {"distance": "1", "persistent": "false"},
        "minecraft:spruce_leaves": {"distance": "1", "persistent": "false"},
        "minecraft:oak_fence_gate": {
            "facing": "north",
            "open": "false",
            "powered": "false",
            "in_wall": "false",
        },
        "minecraft:spruce_fence_gate": {
            "facing": "north",
            "open": "false",
            "powered": "false",
            "in_wall": "false",
        },
        "minecraft:cobblestone_wall": {
            "up": "true",
            "north": "none",
            "south": "none",
            "west": "none",
            "east": "none",
        },
        "minecraft:stone_brick_wall": {
            "up": "true",
            "north": "none",
            "south": "none",
            "west": "none",
            "east": "none",
        },
        "minecraft:torch": {"facing": "up"},
        "minecraft:redstone_torch": {"facing": "up"},
        "minecraft:white_banner": {"rotation": "0"},
        "minecraft:red_banner": {"rotation": "0"},
        "minecraft:glass_pane": {
            "north": "true",
            "south": "true",
            "west": "true",
            "east": "true",
        },
        "minecraft:white_stained_glass_pane": {
            "north": "true",
            "south": "true",
            "west": "true",
            "east": "true",
        },
        "minecraft:grass_block": {"snowy": "false"},
        "minecraft:iron_bars": {
            "north": "true",
            "south": "true",
            "west": "true",
            "east": "true",
        },
    }

    # Ad-hoc check for stairs
    if "stairs" in block_type:
        return {"facing": "north", "half": "bottom", "shape": "straight"}

    return defaults.get(block_type, {})


def remap_problematic_mc_blocks(block_type):
    remapping = {"minecraft:glass_pane": "minecraft:glass"}

    if block_type in remapping:
        return remapping[block_type]
    return block_type


if __name__ == "__main__":
    pass
