```python
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
    """Create a stack of toy blocks with different colors and slight offsets"""
    colors = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0),
              (1.0, 1.0, 0.5), (1.0, 0.5, 1.0), (0.5, 1.0, 1.0)]
    
    # Fixed offsets for deterministic results
    offsets = [(0.02, 0, 0.01), (-0.01, 0, 0.02), (0.01, 0, -0.02), 
               (-0.02, 0, -0.01), (0.015, 0, 0.015), (-0.015, 0, -0.015)]

    def loop_fn(i) -> Shape:
        # Decrease size slightly for each higher block
        size_factor = 1.0 - (i * 0.1)
        block_size = (base_size * size_factor, height, base_size * size_factor)
        color = colors[i % len(colors)]

        block = library_call('toy_block', scale=block_size, color=color)
        # Use fixed offsets instead of random
        offset_idx = i % len(offsets)
        offset = (offsets[offset_idx][0], i * height, offsets[offset_idx][2])
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

    # Wheels - corrected to be horizontal
    wheel_radius = height * 0.3

    def create_wheel(x_pos: float, z_pos: float) -> Shape:
        # Horizontal wheel orientation (along x-axis)
        wheel = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                              shape_kwargs={'radius': wheel_radius,
                                           'p0': (x_pos - width * 0.1, wheel_radius, z_pos),
                                           'p1': (x_pos + width * 0.1, wheel_radius, z_pos)})
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

    # Head - adjusted position to connect better with body
    head = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                         shape_kwargs={'radius': size * 0.3})
    head = transform_shape(head, translation_matrix((0, size * 1.1, 0)))

    # Ears - adjusted position to connect better with head
    ear_radius = size * 0.12
    left_ear = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                             shape_kwargs={'radius': ear_radius})
    right_ear = primitive_call('sphere', color=(0.8, 0.5, 0.3),
                              shape_kwargs={'radius': ear_radius})

    left_ear = transform_shape(left_ear, translation_matrix((-size * 0.25, size * 1.3, 0)))
    right_ear = transform_shape(right_ear, translation_matrix((size * 0.25, size * 1.3, 0)))

    # Arms - adjusted to connect better with body
    arm_radius = size * 0.12
    left_arm = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                             shape_kwargs={'radius': arm_radius,
                                          'p0': (-size * 0.4, size * 0.6, 0),
                                          'p1': (-size * 0.7, size * 0.4, 0)})
    right_arm = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                              shape_kwargs={'radius': arm_radius,
                                           'p0': (size * 0.4, size * 0.6, 0),
                                           'p1': (size * 0.7, size * 0.4, 0)})

    # Legs - adjusted to connect better with body
    leg_radius = size * 0.12
    left_leg = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                             shape_kwargs={'radius': leg_radius,
                                          'p0': (-size * 0.25, size * 0.2, 0),
                                          'p1': (-size * 0.35, -size * 0.2, 0)})
    right_leg = primitive_call('cylinder', color=(0.8, 0.5, 0.3),
                              shape_kwargs={'radius': leg_radius,
                                           'p0': (size * 0.25, size * 0.2, 0),
                                           'p1': (size * 0.35, -size * 0.2, 0)})

    # Eyes and nose - adjusted positions
    left_eye = primitive_call('sphere', color=(0.1, 0.1, 0.1),
                             shape_kwargs={'radius': size * 0.05})
    right_eye = primitive_call('sphere', color=(0.1, 0.1, 0.1),
                              shape_kwargs={'radius': size * 0.05})
    nose = primitive_call('sphere', color=(0.1, 0.1, 0.1),
                         shape_kwargs={'radius': size * 0.06})

    left_eye = transform_shape(left_eye, translation_matrix((-size * 0.15, size * 1.15, -size * 0.25)))
    right_eye = transform_shape(right_eye, translation_matrix((size * 0.15, size * 1.15, -size * 0.25)))
    nose = transform_shape(nose, translation_matrix((0, size * 1.05, -size * 0.28)))

    return concat_shapes(body, head, left_ear, right_ear, left_arm, right_arm,
                         left_leg, right_leg, left_eye, right_eye, nose)

@register()
def play_mat(width: float, length: float, thickness: float) -> Shape:
    """Create a colorful play mat for the children's corner"""
    mat = primitive_call('cube', color=(0.2, 0.8, 0.2),
                        shape_kwargs={'scale': (width, thickness, length)})

    # Add colorful squares pattern - true checkerboard pattern
    squares = []
    num_squares_x = 5
    num_squares_z = 5
    square_width = width / num_squares_x
    square_length = length / num_squares_z

    colors = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0),
              (1.0, 1.0, 0.5), (1.0, 0.5, 1.0)]

    for i in range(num_squares_x):
        for j in range(num_squares_z):
            color = colors[(i + j) % len(colors)]
            square = primitive_call('cube', color=color,
                                   shape_kwargs={'scale': (square_width * 0.95, thickness * 1.1, square_length * 0.95)})
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
def wall_corner() -> Shape:
    """Create a simple wall corner backdrop"""
    # Reduced wall size for better proportions
    wall_height = 2.0
    wall_width = 3.0
    wall_thickness = 0.1

    # Left wall
    left_wall = primitive_call('cube', color=(0.95, 0.95, 0.85),
                              shape_kwargs={'scale': (wall_thickness, wall_height, wall_width)})
    left_wall = transform_shape(left_wall, translation_matrix((-wall_width/2, wall_height/2, 0)))

    # Back wall
    back_wall = primitive_call('cube', color=(0.9, 0.9, 0.8),
                              shape_kwargs={'scale': (wall_width, wall_height, wall_thickness)})
    back_wall = transform_shape(back_wall, translation_matrix((0, wall_height/2, -wall_width/2)))

    return concat_shapes(left_wall, back_wall)

@register()
def small_chair() -> Shape:
    """Create a small children's chair"""
    # Chair dimensions
    seat_width = 0.35
    seat_height = 0.25
    seat_depth = 0.35
    back_height = 0.35
    leg_radius = 0.02

    # Seat
    seat = primitive_call('cube', color=(0.7, 0.4, 0.3),
                         shape_kwargs={'scale': (seat_width, seat_height * 0.2, seat_depth)})
    seat = transform_shape(seat, translation_matrix((0, seat_height, 0)))

    # Back
    back = primitive_call('cube', color=(0.7, 0.4, 0.3),
                         shape_kwargs={'scale': (seat_width, back_height, seat_depth * 0.1)})
    back = transform_shape(back, translation_matrix((0, seat_height + back_height/2, -seat_depth/2 + seat_depth*0.05)))

    # Legs - corrected positioning
    legs = []
    leg_positions = [
        (seat_width/2 - leg_radius, 0, seat_depth/2 - leg_radius),
        (seat_width/2 - leg_radius, 0, -seat_depth/2 + leg_radius),
        (-seat_width/2 + leg_radius, 0, seat_depth/2 - leg_radius),
        (-seat_width/2 + leg_radius, 0, -seat_depth/2 + leg_radius)
    ]

    for pos in leg_positions:
        leg = primitive_call('cylinder', color=(0.6, 0.3, 0.2),
                            shape_kwargs={'radius': leg_radius,
                                         'p0': pos,
                                         'p1': (pos[0], seat_height, pos[2])})
        legs.append(leg)

    return concat_shapes(seat, back, *legs)

@register()
def children_playing_corner() -> Shape:
    """Create a complete children's playing corner with toys, mat, and furniture"""
    # Create play mat as the base - reduced size for better proportions
    mat_thickness = 0.05
    mat_width = 3.0
    mat_length = 3.0
    mat = library_call('play_mat', width=mat_width, length=mat_length, thickness=mat_thickness)
    
    # Create wall corner backdrop - positioned to be visible but not dominating
    walls = library_call('wall_corner')
    walls = transform_shape(walls, translation_matrix((0, 0, 0)))

    # Create toy chest - reduced size for better proportions
    chest_width = 0.6
    chest_height = 0.4
    chest_depth = 0.4
    chest = library_call('toy_chest', width=chest_width, height=chest_height, depth=chest_depth)
    # Position chest properly on the mat
    chest = transform_shape(chest, translation_matrix((1.0, mat_thickness + chest_height/2, -1.0)))

    # Create small chair - positioned to be visible
    chair = library_call('small_chair')
    chair_center = compute_shape_center(chair)
    chair_min = compute_shape_min(chair)
    # Position chair properly on the mat
    chair_y_offset = mat_thickness - chair_min[1]
    chair = transform_shape(chair, translation_matrix((0.8, chair_y_offset, 0.8)))
    chair = transform_shape(chair, rotation_matrix(math.radians(-30), direction=(0, 1, 0), point=chair_center))

    # Create teddy bear - properly positioned on the mat
    teddy_size = 0.3
    teddy = library_call('teddy_bear', size=teddy_size)
    teddy_min = compute_shape_min(teddy)
    # Position teddy to sit properly on the mat
    teddy_y_offset = mat_thickness - teddy_min[1]
    teddy = transform_shape(teddy, translation_matrix((-0.8, teddy_y_offset, -0.8)))
    teddy = transform_shape(teddy, rotation_matrix(math.radians(30), direction=(0, 1, 0), point=compute_shape_center(teddy)))

    # Create toy train - properly positioned on the mat
    train = library_call('toy_train', length=0.4, height=0.15, width=0.15)
    train_min = compute_shape_min(train)
    # Position train to sit properly on the mat
    train_y_offset = mat_thickness - train_min[1]
    train = transform_shape(train, translation_matrix((0.4, train_y_offset, 0.6)))
    train = transform_shape(train, rotation_matrix(math.radians(-45), direction=(0, 1, 0), point=compute_shape_center(train)))

    # Create stack of blocks - properly positioned on the mat
    blocks = library_call('toy_blocks_stack', base_size=0.2, height=0.08, num_blocks=4)
    blocks_min = compute_shape_min(blocks)
    # Position blocks to sit properly on the mat
    blocks_y_offset = mat_thickness - blocks_min[1]
    blocks = transform_shape(blocks, translation_matrix((-0.9, blocks_y_offset, 0.7)))

    # Create scattered toy balls - properly positioned on the mat
    balls = []
    ball_positions = [(-0.4, 0.5), (0.6, -0.5), (-0.6, -0.3), (0.9, 0.2)]
    ball_colors = [(1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0), (1.0, 1.0, 0.3)]
    ball_sizes = [0.12, 0.10, 0.14, 0.11]

    for i, ((x, z), color, size) in enumerate(zip(ball_positions, ball_colors, ball_sizes)):
        ball = library_call('toy_ball', radius=size, color=color)
        # Position ball to sit properly on the mat (y = mat_thickness + radius)
        ball = transform_shape(ball, translation_matrix((x, mat_thickness + size, z)))
        balls.append(ball)

    # Additional toy blocks scattered around - properly positioned on the mat
    scattered_blocks = []
    block_positions = [(0.2, -0.7), (-0.7, 0.3), (0.6, 0.1)]
    block_colors = [(1.0, 0.8, 0.2), (0.2, 0.8, 1.0), (0.8, 0.2, 1.0)]
    block_sizes = [(0.15, 0.15, 0.15), (0.12, 0.12, 0.12), (0.18, 0.09, 0.12)]

    for i, ((x, z), color, size) in enumerate(zip(block_positions, block_colors, block_sizes)):
        block = library_call('toy_block', scale=size, color=color)
        # Position block to sit properly on the mat (y = mat_thickness + height/2)
        block = transform_shape(block, translation_matrix((x, mat_thickness + size[1]/2, z)))
        # Add some rotation for natural look
        block = transform_shape(block, rotation_matrix(math.radians(i * 30),
                                                     direction=(0, 1, 0),
                                                     point=compute_shape_center(block)))
        scattered_blocks.append(block)

    return concat_shapes(
        walls,
        mat,
        chest,
        chair,
        teddy,
        train,
        blocks,
        *balls,
        *scattered_blocks
    )
```