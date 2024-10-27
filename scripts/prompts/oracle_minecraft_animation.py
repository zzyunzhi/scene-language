from helper import *

"""
An animation of Pac-Man being chased by two ghosts.
"""

@register_animation("Animates Pac-Man chasing ghosts across the screen.")
def pacman_chase_animation() -> Generator[Shape, None, None]:
    total_frames = 20  # Number of frames in the animation
    
    for frame in range(total_frames):
        pacman_x = - frame 
        ghost_x_offset = -frame  

        # Move pacman and ghost
        pacman = transform_shape(library_call('pacman'), translation_matrix([pacman_x, 0, 0]))
        ghosts = transform_shape(library_call('ghosts'), translation_matrix([ghost_x_offset, 0, 0]))

        # Export the shape, which is a frame in the animation
        yield concat_shapes(pacman, ghosts)

@register()
def pacman_scene() -> Shape:
    return transform_shape(
        concat_shapes(
            library_call('pacman'),
            library_call('ghosts')
        ),
        translation_matrix([20, 0, 0])
    )

@register()
def pacman() -> Shape:
    return concat_shapes(
        library_call('pacman_shape'),
        library_call('pacman_mouth')
    )

@register()
def pacman_mouth() -> Shape:
    return transform_shape(
        primitive_call('delete_blocks', scale=(6, 2, 1)),
        translation_matrix([0, 4, 0])
    )

@register()
def pacman_shape() -> Shape:
    return primitive_call('set_cuboid', block_type='minecraft:yellow_concrete', scale=(10, 10, 1), fill=False)

@register()
def ghosts() -> Shape:
    colors = ['red', 'lime', 'blue']
    return loop(3, lambda i: transform_shape(
        library_call('ghost', color=colors[i]),
        translation_matrix([15 + i * 12, 0, 0])
    ))

@register()
def ghost(color: str) -> Shape:
    return concat_shapes(
        library_call('ghost_body', color=color),
        library_call('ghost_eyes')
    )

@register()
def ghost_body(color: str) -> Shape:
    block_type = f'minecraft:{color}_concrete'
    return concat_shapes(
        primitive_call('set_cuboid', block_type=block_type, scale=(10, 8, 1), fill=True),
        transform_shape(
            primitive_call('set_cuboid', block_type=block_type, scale=(10, 2, 1), fill=False),
            translation_matrix([0, -2, 0])
        )
    )

@register()
def ghost_eyes() -> Shape:
    eye = primitive_call('set_cuboid', block_type='minecraft:white_concrete', scale=(2, 2, 1), fill=True)
    return concat_shapes(
        transform_shape(eye, translation_matrix([2, 2, 0])),
        transform_shape(eye, translation_matrix([6, 2, 0]))
    )


if __name__ == "__main__":
    from impl import set_seed

    set_seed(1)

    import sys
    import traceback
    import ipdb

    from dsl_utils import animation_func
    from shape_utils import create_hole
    from minecraft_helper import execute_animation
    from pathlib import Path

    name = "pacman_chase_animation"
    save_dir = Path(__file__).parent.parent / 'outputs/example_minedojo' / name
    save_dir.mkdir(parents=True, exist_ok=True)
    frames = list(animation_func())
    execute_animation(frames, save_dir=(save_dir / 'default').as_posix(), description=name)