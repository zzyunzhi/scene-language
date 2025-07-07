```python
from helper import *

"""
children playing corner
"""

# Set a fixed seed for reproducibility
np.random.seed(42)

@register()
def toy_block(scale: P, color: tuple[float, float, float] = (1.0, 0.5, 0.5)) -> Shape:
    """Create a colorful toy block with the given scale and color."""
    return primitive_call('cube', color=color, shape_kwargs={'scale': scale})

@register()
def toy_ball(radius: float, color: tuple[float, float, float] = (0.5, 0.5, 1.0)) -> Shape:
    """Create a colorful toy ball with the given radius and color."""
    return primitive_call('sphere', color=color, shape_kwargs={'radius': radius})

@register()
def toy_blocks_stack(base_size: float, height: float, num_blocks: int) -> Shape:
    """Create a stack of toy blocks with random colors and slight offsets."""
    colors = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0),
              (1.0, 1.0, 0.5), (1.0, 0.5, 1.0), (0.5, 1.0, 1.0)]

    def loop_fn(i) -> Shape:
        # Decrease size more noticeably as we go up
        size_factor = 1.0 - (i * 0.15)
        block_size = (base_size * size_factor, height, base_size * size_factor)

        # Pick a color from our palette
        color_idx = i % len(colors)

        block = primitive_call('cube', color=colors[color_idx],
                              shape_kwargs={'scale': block_size})

        # Add slight random offset for a more natural look
        offset_x = np.random.uniform(-0.05, 0.05) * (i > 0)
        offset_z = np.random.uniform(-0.05, 0.05) * (i > 0)

        # Position the block
        block = transform_shape(block, translation_matrix([offset_x, i * height + height/2, offset_z]))

        return block

    return loop(num_blocks, loop_fn)

@register()
def toy_balls_pile(radius: float, num_balls: int, spread: float) -> Shape:
    """Create a pile of colorful toy balls with realistic stacking."""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
              (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
    
    # Create structured positions for a realistic pile
    positions = []
    
    # Base layer in a circle
    for i in range(min(num_balls // 2, 6)):
        angle = 2 * math.pi * i / min(num_balls // 2, 6)
        pos_x = math.cos(angle) * spread * 0.7
        pos_z = math.sin(angle) * spread * 0.7
        positions.append((pos_x, radius, pos_z))
    
    # Second layer (fewer balls)
    for i in range(min(num_balls // 4, 3)):
        angle = 2 * math.pi * i / min(num_balls // 4, 3)
        pos_x = math.cos(angle) * spread * 0.4
        pos_z = math.sin(angle) * spread * 0.4
        positions.append((pos_x, radius * 2.7, pos_z))
    
    # Top ball
    if num_balls > len(positions):
        positions.append((0, radius * 4.4, 0))
    
    # Fill remaining positions if needed
    while len(positions) < num_balls:
        pos_x = np.random.uniform(-spread * 0.7, spread * 0.7)
        pos_y = radius * 1.5
        pos_z = np.random.uniform(-spread * 0.7, spread * 0.7)
        positions.append((pos_x, pos_y, pos_z))
    
    # Create balls at these positions
    balls = []
    for i, (pos_x, pos_y, pos_z) in enumerate(positions):
        if i >= num_balls:
            break
        color_idx = i % len(colors)
        ball = primitive_call('sphere', color=colors[color_idx],
                           shape_kwargs={'radius': radius})
        ball = transform_shape(ball, translation_matrix([pos_x, pos_y, pos_z]))
        balls.append(ball)
    
    return concat_shapes(*balls)

@register()
def toy_train(length: float, height: float, width: float) -> Shape:
    """Create a simple toy train with a body and wheels."""
    # Train body
    body = primitive_call('cube', color=(1.0, 0.0, 0.0),
                         shape_kwargs={'scale': (length, height, width)})

    # Train cabin - make it more distinct
    cabin_height = height * 1.2
    cabin_length = length * 0.3
    cabin = primitive_call('cube', color=(0.0, 0.0, 1.0),
                          shape_kwargs={'scale': (cabin_length, cabin_height, width * 0.9)})

    cabin_pos_x = (length - cabin_length) / 2
    cabin_pos_y = height + cabin_height / 2
    cabin = transform_shape(cabin, translation_matrix([cabin_pos_x, cabin_pos_y, 0]))

    # Wheels - corrected orientation
    wheel_radius = height * 0.3

    # Front wheels - properly oriented along z-axis
    front_wheel_left = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                                    shape_kwargs={'radius': wheel_radius,
                                                 'p0': (-length/3, -height/2, width/2),
                                                 'p1': (-length/3, -height/2, -width/2)})
    
    front_wheel_right = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                                     shape_kwargs={'radius': wheel_radius,
                                                  'p0': (-length/3, -height/2, -width/2),
                                                  'p1': (-length/3, -height/2, width/2)})

    # Back wheels
    back_wheel_left = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                                   shape_kwargs={'radius': wheel_radius,
                                                'p0': (length/3, -height/2, width/2),
                                                'p1': (length/3, -height/2, -width/2)})
    
    back_wheel_right = primitive_call('cylinder', color=(0.2, 0.2, 0.2),
                                    shape_kwargs={'radius': wheel_radius,
                                                 'p0': (length/3, -height/2, -width/2),
                                                 'p1': (length/3, -height/2, width/2)})

    return concat_shapes(body, cabin, front_wheel_left, front_wheel_right, back_wheel_left, back_wheel_right)

@register()
def play_mat(width: float, length: float, thickness: float) -> Shape:
    """Create a colorful play mat for the children's corner."""
    mat = primitive_call('cube', color=(0.2, 0.8, 0.2),
                        shape_kwargs={'scale': (width, thickness, length)})
    return mat

@register()
def toy_shelf(width: float, height: float, depth: float) -> Shape:
    """Create a simple toy shelf with multiple compartments."""
    # Main shelf body
    shelf_body = primitive_call('cube', color=(0.8, 0.7, 0.6),
                               shape_kwargs={'scale': (width, height, depth)})

    # Shelf dividers (horizontal) - make them thicker
    num_shelves = 3
    shelf_thickness = height * 0.08

    shelves = []
    for i in range(1, num_shelves):
        y_pos = -height/2 + (height * i / num_shelves)
        shelf = primitive_call('cube', color=(0.7, 0.6, 0.5),
                              shape_kwargs={'scale': (width - 0.05, shelf_thickness, depth - 0.05)})
        shelf = transform_shape(shelf, translation_matrix([0, y_pos, 0]))
        shelves.append(shelf)

    # Vertical dividers - make them thicker
    num_dividers = 2
    divider_thickness = width * 0.08

    dividers = []
    for i in range(1, num_dividers):
        x_pos = -width/2 + (width * i / num_dividers)
        divider = primitive_call('cube', color=(0.7, 0.6, 0.5),
                                shape_kwargs={'scale': (divider_thickness, height - 0.05, depth - 0.05)})
        divider = transform_shape(divider, translation_matrix([x_pos, 0, 0]))
        dividers.append(divider)

    return concat_shapes(shelf_body, *shelves, *dividers)

@register()
def stuffed_animal(base_size: float, color: tuple[float, float, float]) -> Shape:
    """Create a simple stuffed animal toy using spheres."""
    # Body
    body = primitive_call('sphere', color=color,
                         shape_kwargs={'radius': base_size * 0.6})

    # Head
    head = primitive_call('sphere', color=color,
                         shape_kwargs={'radius': base_size * 0.4})
    head = transform_shape(head, translation_matrix([0, base_size * 0.7, 0]))

    # Ears
    ear_left = primitive_call('sphere', color=color,
                             shape_kwargs={'radius': base_size * 0.15})
    ear_left = transform_shape(ear_left, translation_matrix([base_size * 0.3, base_size * 1.1, 0]))

    ear_right = primitive_call('sphere', color=color,
                              shape_kwargs={'radius': base_size * 0.15})
    ear_right = transform_shape(ear_right, translation_matrix([-base_size * 0.3, base_size * 1.1, 0]))

    # Arms
    arm_left = primitive_call('sphere', color=color,
                             shape_kwargs={'radius': base_size * 0.2})
    arm_left = transform_shape(arm_left, translation_matrix([base_size * 0.6, 0, 0]))

    arm_right = primitive_call('sphere', color=color,
                              shape_kwargs={'radius': base_size * 0.2})
    arm_right = transform_shape(arm_right, translation_matrix([-base_size * 0.6, 0, 0]))

    # Legs
    leg_left = primitive_call('sphere', color=color,
                             shape_kwargs={'radius': base_size * 0.25})
    leg_left = transform_shape(leg_left, translation_matrix([base_size * 0.3, -base_size * 0.6, 0]))

    leg_right = primitive_call('sphere', color=color,
                              shape_kwargs={'radius': base_size * 0.25})
    leg_right = transform_shape(leg_right, translation_matrix([-base_size * 0.3, -base_size * 0.6, 0]))

    return concat_shapes(body, head, ear_left, ear_right, arm_left, arm_right, leg_left, leg_right)

@register()
def small_rug(width: float, length: float, thickness: float, color: tuple[float, float, float]) -> Shape:
    """Create a small decorative rug for a play zone."""
    rug = primitive_call('cube', color=color,
                        shape_kwargs={'scale': (width, thickness, length)})
    return rug

@register()
def children_playing_corner() -> Shape:
    """Create a complete children's playing corner with toys and furniture."""
    # Mat thickness constant for reference
    mat_thickness = 0.05

    # Create the play mat as the base
    mat = library_call('play_mat', width=4.0, length=4.0, thickness=mat_thickness)

    # Create play zones with small rugs
    building_rug = library_call('small_rug', width=1.2, length=1.2, thickness=0.02, color=(0.9, 0.7, 0.3))
    building_rug = transform_shape(building_rug, translation_matrix([-1.2, mat_thickness/2 + 0.01, -1.2]))

    reading_rug = library_call('small_rug', width=1.2, length=1.2, thickness=0.02, color=(0.7, 0.3, 0.9))
    reading_rug = transform_shape(reading_rug, translation_matrix([1.2, mat_thickness/2 + 0.01, 1.2]))

    # Add a toy shelf - fixed positioning to be fully on the mat
    shelf = library_call('toy_shelf', width=1.2, height=1.0, depth=0.35)
    shelf_height = 1.0
    shelf = transform_shape(shelf, translation_matrix([1.2, shelf_height/2 + mat_thickness/2, -1.2]))

    # Add toy blocks in the building zone
    blocks = library_call('toy_blocks_stack', base_size=0.2, height=0.1, num_blocks=5)
    blocks = transform_shape(blocks, translation_matrix([-1.2, mat_thickness, -1.2]))

    # Add a pile of toy balls - improved stacking
    balls = library_call('toy_balls_pile', radius=0.1, num_balls=8, spread=0.3)
    balls = transform_shape(balls, translation_matrix([1.0, mat_thickness, 1.0]))

    # Add a toy train - fixed positioning to sit on the mat
    train = library_call('toy_train', length=0.5, height=0.15, width=0.2)
    train = transform_shape(train, translation_matrix([-1.0, 0.15/2 + mat_thickness/2, 0.5]))

    # Add some stuffed animals - increased size for better proportion
    teddy = library_call('stuffed_animal', base_size=0.4, color=(0.8, 0.6, 0.4))
    teddy = transform_shape(teddy, translation_matrix([0.5, 0.4 * 0.6 + mat_thickness/2, -1.0]))

    bunny = library_call('stuffed_animal', base_size=0.35, color=(0.9, 0.9, 0.9))
    bunny = transform_shape(bunny, translation_matrix([-0.8, 0.35 * 0.6 + mat_thickness/2, -0.5]))

    # Add some scattered blocks - with improved collision avoidance
    scattered_blocks = []
    block_positions = [(-0.7, 0.8), (0.7, -0.8), (-0.3, -0.3), (0.3, 0.3), (1.5, 0.0)]
    occupied_positions = []

    for i, (pos_x, pos_z) in enumerate(block_positions):
        # Check if position is already occupied
        is_occupied = any(math.sqrt((pos_x - x)**2 + (pos_z - z)**2) < 0.2 for x, z in occupied_positions)
        if is_occupied:
            # Adjust position
            pos_x += 0.2
            pos_z += 0.2
        
        occupied_positions.append((pos_x, pos_z))
        
        size_x = np.random.uniform(0.1, 0.15)
        size_y = np.random.uniform(0.1, 0.15)
        size_z = np.random.uniform(0.1, 0.15)

        color_r = np.random.uniform(0.5, 1.0)
        color_g = np.random.uniform(0.5, 1.0)
        color_b = np.random.uniform(0.5, 1.0)

        block = primitive_call('cube', color=(color_r, color_g, color_b),
                              shape_kwargs={'scale': (size_x, size_y, size_z)})

        # Position blocks on the mat
        block = transform_shape(block, translation_matrix([pos_x, size_y/2 + mat_thickness/2, pos_z]))
        scattered_blocks.append(block)

    # Add some books on the shelf - properly positioned on shelf dividers
    books = []
    # Position books to sit properly on the shelf dividers
    book_positions = [(1.2, 0.55, -1.4), (1.4, 0.55, -1.4), (1.6, 0.55, -1.4)]
    book_colors = [(0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.2, 0.2, 0.9)]

    for i, (pos_x, pos_y, pos_z) in enumerate(book_positions):
        book = primitive_call('cube', color=book_colors[i % len(book_colors)],
                             shape_kwargs={'scale': (0.1, 0.2, 0.15)})
        book = transform_shape(book, translation_matrix([pos_x, pos_y + mat_thickness/2, pos_z]))
        books.append(book)

    return concat_shapes(
        mat,
        building_rug,
        reading_rug,
        shelf,
        blocks,
        balls,
        train,
        teddy,
        bunny,
        *scattered_blocks,
        *books
    )
```