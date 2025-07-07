```python
from helper import *

"""
children playing corner
"""

# Set random seed for reproducibility
np.random.seed(42)

@register()
def toy_block(scale: P, color: tuple[float, float, float] = (1.0, 0.5, 0.0)) -> Shape:
    """Create a toy block with given scale and color"""
    return primitive_call('cube', color=color, shape_kwargs={'scale': scale})

@register()
def ball(radius: float, color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Shape:
    """Create a ball with given radius and color"""
    return primitive_call('sphere', color=color, shape_kwargs={'radius': radius})

@register()
def toy_blocks_stack(num_blocks: int = 5) -> Shape:
    """Create a stack of toy blocks with random colors and slight offsets"""
    colors = [(1.0, 0.5, 0.0), (0.0, 0.7, 0.3), (0.3, 0.3, 1.0),
              (1.0, 0.8, 0.0), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8)]

    def loop_fn(i) -> Shape:
        block_size = (0.15, 0.05, 0.1)
        color_idx = i % len(colors)
        block = library_call('toy_block', scale=block_size, color=colors[color_idx])
        offset_x = np.random.uniform(-0.03, 0.03)
        offset_z = np.random.uniform(-0.03, 0.03)
        block = transform_shape(block, translation_matrix([offset_x, i * block_size[1], offset_z]))
        return block

    return loop(num_blocks, loop_fn)

@register()
def toy_blocks_pyramid(levels: int = 4) -> Shape:
    """Create a pyramid of toy blocks"""
    blocks = []

    for level in range(levels):
        num_blocks_in_level = levels - level
        for i in range(num_blocks_in_level):
            color = (np.random.uniform(0.3, 1.0),
                     np.random.uniform(0.3, 1.0),
                     np.random.uniform(0.3, 1.0))
            block = library_call('toy_block', scale=(0.1, 0.05, 0.1), color=color)

            # Position blocks in a row for this level
            offset_x = (i - (num_blocks_in_level - 1) / 2) * 0.12
            offset_y = level * 0.06
            offset_z = 0

            block = transform_shape(block, translation_matrix([offset_x, offset_y, offset_z]))
            blocks.append(block)

    return concat_shapes(*blocks)

@register()
def ball_pile(num_balls: int = 6) -> Shape:
    """Create a pile of colorful balls"""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
              (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]

    def loop_fn(i) -> Shape:
        radius = np.random.uniform(0.04, 0.07)
        color_idx = i % len(colors)
        ball = library_call('ball', radius=radius, color=colors[color_idx])

        # Arrange balls in a tighter pile for more cohesion
        if i < 3:  # First layer
            angle = i * (2 * math.pi / 3)
            offset_x = 0.06 * math.cos(angle)  # Reduced from 0.08
            offset_z = 0.06 * math.sin(angle)  # Reduced from 0.08
            offset_y = radius
        else:  # Second layer
            angle = (i-3) * (2 * math.pi / 3) + (math.pi/3)
            offset_x = 0.03 * math.cos(angle)  # Reduced from 0.05
            offset_z = 0.03 * math.sin(angle)  # Reduced from 0.05
            offset_y = 0.10 + radius  # Reduced from 0.12

        return transform_shape(ball, translation_matrix([offset_x, offset_y, offset_z]))

    return loop(num_balls, loop_fn)

@register()
def toy_train(length: float = 0.3) -> Shape:
    """Create a simple toy train"""
    # Train body
    body = primitive_call('cube', color=(0.7, 0.0, 0.0), shape_kwargs={'scale': (length, 0.1, 0.12)})

    # Train cabin
    cabin = primitive_call('cube', color=(0.8, 0.0, 0.0),
                          shape_kwargs={'scale': (0.12, 0.15, 0.12)})
    cabin = transform_shape(cabin, translation_matrix([length/4, 0.125, 0]))

    # Wheels - fixed to be perpendicular to the train body (along x-axis)
    wheels = []
    wheel_positions = [(-length/3, -0.05, 0.07), (-length/3, -0.05, -0.07),
                       (length/3, -0.05, 0.07), (length/3, -0.05, -0.07)]

    for pos in wheel_positions:
        wheel = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                              shape_kwargs={'radius': 0.03, 'p0': (pos[0]-0.02, pos[1], pos[2]),
                                           'p1': (pos[0]+0.02, pos[1], pos[2])})
        wheels.append(wheel)

    # Chimney - ensure it's vertical (along y-axis)
    chimney = primitive_call('cylinder', color=(0.3, 0.3, 0.3),
                            shape_kwargs={'radius': 0.02, 'p0': (length/4, 0.2, 0),
                                         'p1': (length/4, 0.3, 0)})

    return concat_shapes(body, cabin, chimney, *wheels)

@register()
def play_mat() -> Shape:
    """Create a colorful play mat"""
    return primitive_call('cube', color=(0.0, 0.6, 0.2),
                         shape_kwargs={'scale': (1.5, 0.02, 1.5)})

@register()
def toy_shelf() -> Shape:
    """Create a simple toy shelf with proper structure"""
    # Back panel
    back_panel = primitive_call('cube', color=(0.8, 0.7, 0.6),
                              shape_kwargs={'scale': (0.8, 0.6, 0.05)})
    back_panel = transform_shape(back_panel, translation_matrix([0, 0, -0.125]))

    # Side panels
    left_panel = primitive_call('cube', color=(0.8, 0.7, 0.6),
                              shape_kwargs={'scale': (0.05, 0.6, 0.3)})
    left_panel = transform_shape(left_panel, translation_matrix([-0.375, 0, 0]))

    right_panel = primitive_call('cube', color=(0.8, 0.7, 0.6),
                               shape_kwargs={'scale': (0.05, 0.6, 0.3)})
    right_panel = transform_shape(right_panel, translation_matrix([0.375, 0, 0]))

    # Shelves
    shelf1 = primitive_call('cube', color=(0.75, 0.65, 0.55),
                          shape_kwargs={'scale': (0.75, 0.02, 0.28)})
    shelf1 = transform_shape(shelf1, translation_matrix([0, -0.1, 0]))

    shelf2 = primitive_call('cube', color=(0.75, 0.65, 0.55),
                          shape_kwargs={'scale': (0.75, 0.02, 0.28)})
    shelf2 = transform_shape(shelf2, translation_matrix([0, 0.1, 0]))

    # Add some decorative elements to the shelf
    book1 = primitive_call('cube', color=(0.9, 0.2, 0.2),
                          shape_kwargs={'scale': (0.08, 0.12, 0.04)})
    book1 = transform_shape(book1, translation_matrix([-0.3, 0.16, 0]))

    book2 = primitive_call('cube', color=(0.2, 0.2, 0.9),
                          shape_kwargs={'scale': (0.08, 0.10, 0.04)})
    book2 = transform_shape(book2, translation_matrix([-0.2, 0.15, 0]))

    toy_box = primitive_call('cube', color=(1.0, 0.8, 0.0),
                           shape_kwargs={'scale': (0.15, 0.08, 0.15)})
    toy_box = transform_shape(toy_box, translation_matrix([0.3, 0.14, 0]))

    return concat_shapes(back_panel, left_panel, right_panel, shelf1, shelf2, book1, book2, toy_box)

@register()
def teddy_bear() -> Shape:
    """Create a teddy bear with proper scaling"""
    # Reduced scale factor for better proportions
    scale_factor = 0.9

    # Body
    body = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.15 * scale_factor})
    body = transform_shape(body, translation_matrix([0, 0.15 * scale_factor, 0]))

    # Head
    head = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.1 * scale_factor})
    head = transform_shape(head, translation_matrix([0, 0.35 * scale_factor, 0]))

    # Ears
    ear1 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.03 * scale_factor})
    ear1 = transform_shape(ear1, translation_matrix([0.08 * scale_factor, 0.43 * scale_factor, 0]))

    ear2 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.03 * scale_factor})
    ear2 = transform_shape(ear2, translation_matrix([-0.08 * scale_factor, 0.43 * scale_factor, 0]))

    # Arms - fix transformation order: scale first, then translate
    arm1 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    arm1 = transform_shape(arm1, scale_matrix(1.5, (0, 0, 0)))
    arm1 = transform_shape(arm1, translation_matrix([0.2 * scale_factor, 0.15 * scale_factor, 0]))

    arm2 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    arm2 = transform_shape(arm2, scale_matrix(1.5, (0, 0, 0)))
    arm2 = transform_shape(arm2, translation_matrix([-0.2 * scale_factor, 0.15 * scale_factor, 0]))

    # Legs - fix transformation order: scale first, then translate
    leg1 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    leg1 = transform_shape(leg1, scale_matrix(1.5, (0, 0, 0)))
    leg1 = transform_shape(leg1, translation_matrix([0.1 * scale_factor, -0.05 * scale_factor, 0]))

    leg2 = primitive_call('sphere', color=(0.6, 0.4, 0.2),
                         shape_kwargs={'radius': 0.05 * scale_factor})
    leg2 = transform_shape(leg2, scale_matrix(1.5, (0, 0, 0)))
    leg2 = transform_shape(leg2, translation_matrix([-0.1 * scale_factor, -0.05 * scale_factor, 0]))

    # Eyes
    eye1 = primitive_call('sphere', color=(0.0, 0.0, 0.0),
                         shape_kwargs={'radius': 0.015 * scale_factor})
    eye1 = transform_shape(eye1, translation_matrix([0.05 * scale_factor, 0.38 * scale_factor, -0.08 * scale_factor]))

    eye2 = primitive_call('sphere', color=(0.0, 0.0, 0.0),
                         shape_kwargs={'radius': 0.015 * scale_factor})
    eye2 = transform_shape(eye2, translation_matrix([-0.05 * scale_factor, 0.38 * scale_factor, -0.08 * scale_factor]))

    # Nose
    nose = primitive_call('sphere', color=(0.3, 0.2, 0.1),
                         shape_kwargs={'radius': 0.02 * scale_factor})
    nose = transform_shape(nose, translation_matrix([0, 0.33 * scale_factor, -0.09 * scale_factor]))

    return concat_shapes(body, head, ear1, ear2, arm1, arm2, leg1, leg2, eye1, eye2, nose)

@register()
def small_rug() -> Shape:
    """Create a small decorative rug for the play area"""
    rug = primitive_call('cube', color=(0.9, 0.3, 0.3),
                        shape_kwargs={'scale': (0.6, 0.01, 0.4)})
    return rug

@register()
def children_playing_corner() -> Shape:
    """Create a children's playing corner with toys and play area"""
    # Create the play mat as the base
    mat = library_call('play_mat')
    mat_height = 0.01  # Half height of the mat

    # Add a small decorative rug
    rug = library_call('small_rug')
    rug = transform_shape(rug, translation_matrix([0.2, mat_height + 0.005, 0.2]))

    # Add toy blocks in different arrangements - fix positioning to rest on mat
    blocks_stack = library_call('toy_blocks_stack')
    blocks_stack = transform_shape(blocks_stack, translation_matrix([0.4, mat_height + 0.025, 0.3]))

    blocks_pyramid = library_call('toy_blocks_pyramid')
    blocks_pyramid = transform_shape(blocks_pyramid, translation_matrix([-0.4, mat_height + 0.025, 0.4]))

    # Add balls
    balls = library_call('ball_pile')
    balls = transform_shape(balls, translation_matrix([0.3, mat_height + 0.04, -0.3]))

    # Improve toy train visibility
    train = library_call('toy_train')
    train = transform_shape(train, translation_matrix([-0.3, mat_height + 0.05, 0.0]))
    train = transform_shape(train, rotation_matrix(math.pi/4, direction=(0, 1, 0), point=(-0.3, mat_height + 0.05, 0.0)))

    # Fix shelf positioning to rest properly on the floor
    shelf = library_call('toy_shelf')
    shelf_height = 0.6  # Height of the shelf
    shelf = transform_shape(shelf, translation_matrix([0, shelf_height/2 + mat_height, -0.6]))

    # Add teddy bear with fixed proportions and positioning
    bear = library_call('teddy_bear')
    bear = transform_shape(bear, translation_matrix([-0.5, mat_height + 0.15, -0.4]))
    bear = transform_shape(bear, rotation_matrix(math.pi/6, direction=(0, 1, 0), point=(-0.5, mat_height + 0.15, -0.4)))

    # Create some additional balls scattered around with specific positions
    ball_positions = [(0.1, 0.04, 0.1), (-0.3, 0.04, -0.1), (0.5, 0.04, -0.5), (-0.6, 0.04, 0.2)]
    scattered_balls = []
    for i, pos in enumerate(ball_positions):
        ball_color = (np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0))
        ball = library_call('ball', radius=0.04, color=ball_color)
        # Adjust y-coordinate to rest on mat
        scattered_balls.append(transform_shape(ball, translation_matrix([pos[0], mat_height + pos[1], pos[2]])))

    # Add a toy box on the floor
    toy_box = primitive_call('cube', color=(0.3, 0.7, 0.9),
                            shape_kwargs={'scale': (0.2, 0.15, 0.2)})
    toy_box = transform_shape(toy_box, translation_matrix([0.6, mat_height + 0.075, 0.5]))

    return concat_shapes(
        mat,
        rug,
        blocks_stack,
        blocks_pyramid,
        balls,
        train,
        shelf,
        bear,
        toy_box,
        *scattered_balls
    )
```