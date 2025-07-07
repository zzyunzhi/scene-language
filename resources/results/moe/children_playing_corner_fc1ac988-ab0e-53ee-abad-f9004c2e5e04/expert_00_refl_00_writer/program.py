from helper import *

"""
children playing corner
"""

@register()
def toy_block(scale: P, color: tuple[float, float, float] = (1.0, 0.5, 0.5)) -> Shape:
    """Create a colorful toy block with given scale and color"""
    return primitive_call('cube', color=color, shape_kwargs={'scale': scale})

@register()
def toy_ball(radius: float, color: tuple[float, float, float] = (0.3, 0.7, 1.0)) -> Shape:
    """Create a colorful toy ball with given radius and color"""
    return primitive_call('sphere', color=color, shape_kwargs={'radius': radius})

@register()
def toy_blocks_stack(base_size: float, height: float, num_blocks: int) -> Shape:
    """Create a stack of toy blocks with random colors and slight offsets"""
    colors = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0),
              (1.0, 1.0, 0.5), (1.0, 0.5, 1.0), (0.5, 1.0, 1.0)]

    def loop_fn(i) -> Shape:
        # Decrease size slightly for each higher block
        size_factor = 1.0 - (i * 0.1)
        block_size = (base_size * size_factor, height, base_size * size_factor)
        color = colors[i % len(colors)]

        block = library_call('toy_block', scale=block_size, color=color)
        # Add slight random offset for realistic stacking
        offset = (np.random.uniform(-0.05, 0.05), i * height, np.random.uniform(-0.05, 0.05))
        return transform_shape(block, translation_matrix(offset))

    return loop(num_blocks, loop_fn)

@register()
def toy_train(length: float, height: float, width: float) -> Shape:
    """Create a simple toy train with a body and wheels"""
    # Train body
    body = primitive_call('cube', color=(1.0, 0.2, 0.2),
                         shape_kwargs={'scale': (length, height, width)})

    # Train cabin
    cabin_height = height * 0.8
    cabin = primitive_call('cube', color=(0.2, 0.2, 0.8),
                          shape_kwargs={'scale': (length * 0.3, cabin_height, width)})
    cabin = transform_shape(cabin, translation_matrix((length * 0.25, height/2 + cabin_height/2, 0)))

    # Wheels
    wheel_radius = height * 0.3

    def create_wheel(x_pos: float, z_pos: float) -> Shape:
        wheel = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                              shape_kwargs={'radius': wheel_radius,
                                           'p0': (x_pos, wheel_radius, z_pos),
                                           'p1': (x_pos, wheel_radius, z_pos + width * 0.2)})
        return wheel

    wheels = concat_shapes(
        create_wheel(-length * 0.3, -width * 0.3),
        create_wheel(-length * 0.3, width * 0.3),
        create_wheel(length * 0.3, -width * 0.3),
        create_wheel(length * 0.3, width * 0.3)
    )

    return concat_shapes(body, cabin, wheels)

@register()
def teddy_bear(size: float) -> Shape:
    """Create a simple teddy bear with head, body, ears, and limbs"""
    # Body
    body = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                         shape_kwargs={'radius': size * 0.5})
    body = transform_shape(body, translation_matrix((0, size * 0.5, 0)))

    # Head
    head = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                         shape_kwargs={'radius': size * 0.3})
    head = transform_shape(head, translation_matrix((0, size * 1.2, 0)))

    # Ears
    ear_radius = size * 0.15
    left_ear = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                             shape_kwargs={'radius': ear_radius})
    right_ear = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                              shape_kwargs={'radius': ear_radius})

    left_ear = transform_shape(left_ear, translation_matrix((-size * 0.25, size * 1.5, 0)))
    right_ear = transform_shape(right_ear, translation_matrix((size * 0.25, size * 1.5, 0)))

    # Arms
    arm_radius = size * 0.15
    left_arm = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                             shape_kwargs={'radius': arm_radius,
                                          'p0': (-size * 0.5, size * 0.6, 0),
                                          'p1': (-size * 0.8, size * 0.4, 0)})
    right_arm = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                              shape_kwargs={'radius': arm_radius,
                                           'p0': (size * 0.5, size * 0.6, 0),
                                           'p1': (size * 0.8, size * 0.4, 0)})

    # Legs
    leg_radius = size * 0.15
    left_leg = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                             shape_kwargs={'radius': leg_radius,
                                          'p0': (-size * 0.3, size * 0.1, 0),
                                          'p1': (-size * 0.4, -size * 0.3, 0)})
    right_leg = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                              shape_kwargs={'radius': leg_radius,
                                           'p0': (size * 0.3, size * 0.1, 0),
                                           'p1': (size * 0.4, -size * 0.3, 0)})

    # Eyes and nose
    left_eye = primitive_call('sphere', color=(0.1, 0.1, 0.1),
                             shape_kwargs={'radius': size * 0.05})
    right_eye = primitive_call('sphere', color=(0.1, 0.1, 0.1),
                              shape_kwargs={'radius': size * 0.05})
    nose = primitive_call('sphere', color=(0.1, 0.1, 0.1),
                         shape_kwargs={'radius': size * 0.07})

    left_eye = transform_shape(left_eye, translation_matrix((-size * 0.15, size * 1.25, -size * 0.25)))
    right_eye = transform_shape(right_eye, translation_matrix((size * 0.15, size * 1.25, -size * 0.25)))
    nose = transform_shape(nose, translation_matrix((0, size * 1.15, -size * 0.28)))

    return concat_shapes(body, head, left_ear, right_ear, left_arm, right_arm,
                         left_leg, right_leg, left_eye, right_eye, nose)

@register()
def play_mat(width: float, length: float, thickness: float) -> Shape:
    """Create a colorful play mat for the children's corner"""
    mat = primitive_call('cube', color=(0.2, 0.8, 0.2),
                        shape_kwargs={'scale': (width, thickness, length)})

    # Add colorful squares pattern
    squares = []
    num_squares_x = 5
    num_squares_z = 5
    square_width = width / num_squares_x
    square_length = length / num_squares_z

    colors = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0),
              (1.0, 1.0, 0.5), (1.0, 0.5, 1.0)]

    for i in range(num_squares_x):
        for j in range(num_squares_z):
            if (i + j) % 2 == 0:  # Checkerboard pattern
                color = colors[(i + j) % len(colors)]
                square = primitive_call('cube', color=color,
                                       shape_kwargs={'scale': (square_width * 0.9, thickness * 1.1, square_length * 0.9)})
                x_pos = -width/2 + square_width/2 + i * square_width
                z_pos = -length/2 + square_length/2 + j * square_length
                square = transform_shape(square, translation_matrix((x_pos, thickness * 0.05, z_pos)))
                squares.append(square)

    return concat_shapes(mat, *squares)

@register()
def toy_chest(width: float, height: float, depth: float) -> Shape:
    """Create a toy chest to store toys"""
    # Main box
    box = primitive_call('cube', color=(0.8, 0.6, 0.4),
                        shape_kwargs={'scale': (width, height, depth)})

    # Lid
    lid_height = height * 0.1
    lid = primitive_call('cube', color=(0.9, 0.7, 0.5),
                        shape_kwargs={'scale': (width * 1.05, lid_height, depth * 1.05)})
    lid = transform_shape(lid, translation_matrix((0, height/2 + lid_height/2, 0)))

    # Handle
    handle = primitive_call('cylinder', color=(0.7, 0.5, 0.3),
                           shape_kwargs={'radius': height * 0.05,
                                        'p0': (0, height/2 + lid_height + height * 0.05, -depth * 0.25),
                                        'p1': (0, height/2 + lid_height + height * 0.05, depth * 0.25)})

    return concat_shapes(box, lid, handle)

@register()
def children_playing_corner() -> Shape:
    """Create a complete children's playing corner with toys, mat, and furniture"""
    # Create play mat as the base
    mat = library_call('play_mat', width=4.0, length=4.0, thickness=0.05)

    # Create toy chest
    chest = library_call('toy_chest', width=1.0, height=0.8, depth=0.7)
    chest = transform_shape(chest, translation_matrix((1.5, 0.4, -1.5)))

    # Create teddy bear
    teddy = library_call('teddy_bear', size=0.4)
    teddy = transform_shape(teddy, translation_matrix((-1.0, 0.05, -1.0)))
    teddy = transform_shape(teddy, rotation_matrix(math.radians(30), direction=(0, 1, 0), point=compute_shape_center(teddy)))

    # Create toy train
    train = library_call('toy_train', length=0.6, height=0.2, width=0.2)
    train = transform_shape(train, translation_matrix((0.5, 0.1, 0.8)))
    train = transform_shape(train, rotation_matrix(math.radians(-45), direction=(0, 1, 0), point=compute_shape_center(train)))

    # Create stack of blocks
    blocks = library_call('toy_blocks_stack', base_size=0.3, height=0.1, num_blocks=5)
    blocks = transform_shape(blocks, translation_matrix((-1.2, 0.05, 1.0)))

    # Create scattered toy balls
    balls = []
    ball_positions = [(-0.5, 0.15, 0.7), (0.8, 0.15, -0.6), (-0.7, 0.15, -0.3), (1.2, 0.15, 0.3)]
    ball_colors = [(1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0), (1.0, 1.0, 0.3)]
    ball_sizes = [0.15, 0.12, 0.18, 0.14]

    for i, (pos, color, size) in enumerate(zip(ball_positions, ball_colors, ball_sizes)):
        ball = library_call('toy_ball', radius=size, color=color)
        ball = transform_shape(ball, translation_matrix(pos))
        balls.append(ball)

    # Additional toy blocks scattered around
    scattered_blocks = []
    block_positions = [(0.3, 0.1, -0.8), (-0.8, 0.1, 0.4), (0.7, 0.1, 0.2)]
    block_colors = [(1.0, 0.8, 0.2), (0.2, 0.8, 1.0), (0.8, 0.2, 1.0)]
    block_sizes = [(0.2, 0.2, 0.2), (0.15, 0.15, 0.15), (0.25, 0.1, 0.15)]

    for i, (pos, color, size) in enumerate(zip(block_positions, block_colors, block_sizes)):
        block = library_call('toy_block', scale=size, color=color)
        block = transform_shape(block, translation_matrix(pos))
        # Add some rotation for natural look
        block = transform_shape(block, rotation_matrix(math.radians(i * 30),
                                                     direction=(0, 1, 0),
                                                     point=compute_shape_center(block)))
        scattered_blocks.append(block)

    return concat_shapes(
        mat,
        chest,
        teddy,
        train,
        blocks,
        *balls,
        *scattered_blocks
    )