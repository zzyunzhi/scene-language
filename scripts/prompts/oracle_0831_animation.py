from helper import *

"""
three ghosts chasing a yellow pacman
"""


@register()
def pacman() -> Shape:
    return primitive_call('cube', color=(1, 1, 0), scale=.8)


@register()
def ghosts() -> Shape:
    return loop(3, lambda i: transform_shape(
        library_call('ghost', color=(i / 3, 1 - i / 3, 1 - i / 3)),
        translation_matrix([i, 0, 0])
    ))


@register()
def ghost(color) -> Shape:
    return primitive_call('sphere', color=color, scale=.8)


@register_animation()
def pacman_chase_animation() -> Generator[Shape, None, None]:
    # an animated scene
    total_frames = 4  # Number of frames in the animation

    for frame in range(total_frames):
        pacman_x = - frame / total_frames
        ghost_x_offset = - 2 * frame / total_frames

        # Move pacman and ghost
        pacman = transform_shape(library_call('pacman'), translation_matrix([pacman_x, 0, 0]))
        ghosts = transform_shape(library_call('ghosts'), translation_matrix([2 + ghost_x_offset, 0, 0]))

        # Export the shape, which is a frame in the animation
        yield concat_shapes(pacman, ghosts)
if __name__ == "__main__":
    from pathlib import Path
    from impl_utils import run
    from impl_preset import core
    from tu.loggers.utils import print_vcv_url
    root = Path(__file__).parent.parent
    save_dir = root / 'outputs' / Path(__file__).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    print_vcv_url(save_dir.as_posix())
    core(engine_modes=[], overwrite=True, save_dir=save_dir.as_posix())
    exit()
