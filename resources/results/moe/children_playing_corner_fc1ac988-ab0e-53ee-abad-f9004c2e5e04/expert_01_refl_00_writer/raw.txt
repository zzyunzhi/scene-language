```python
from helper import *

"""
children playing corner
"""

@register()
def toy_block(scale: P, color: tuple[float, float, float] = (1.0, 0.5, 0.0)) -> Shape:
    """Create a toy block with given scale and color"""
    return primitive_call('cube', color=color, shape_kwargs={'scale': scale})

@register()
def ball(radius: float, color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Shape:
    """Create a ball with given radius and color"""
    return primitive_call('sphere', color=color, shape_kwargs={'radius': radius})

@register()
def toy_blocks_stack(base_size: float, height: float, num_blocks: int) -> Shape:
    """Create a stack of toy blocks with random colors and slight offsets"""
    colors = [(1.0, 0.5, 0.0), (0.0, 0.7, 0.3), (0.3, 0.3, 1.0), (1.0, 0.8, 0.0), (0.8, 0.2, 0.8)]
    
    def loop_fn(i) -> Shape:
        # Randomize block size slightly
        size_variation = np.random.uniform(0.8, 1.0)
        block_size = (base_size * size_variation, height, base_size * size_variation)
        
        # Select random color
        color = colors[i % len(colors)]
        
        # Create block
        block = library_call('toy_block', scale=block_size, color=color)
        
        # Add random offset and rotation
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_z = np.random.uniform(-0.05, 0.05)
        y_pos = i * height
        
        # Transform block
        block = transform_shape(block, translation_matrix([offset_x, y_pos, offset_z]))
        block_center = compute_shape_center(block)
        rotation_angle = np.random.uniform(-0.2, 0.2)
        return transform_shape(block, rotation_matrix(rotation_angle, direction=(0, 1, 0), point=block_center))
    
    return loop(num_blocks, loop_fn)

@register()
def toy_balls_pile(num_balls: int, radius_range: tuple[float, float] = (0.05, 0.1)) -> Shape:
    """Create a pile of colorful balls with random sizes and positions"""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 
              (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
    
    def loop_fn(i) -> Shape:
        # Random radius
        radius = np.random.uniform(radius_range[0], radius_range[1])
        
        # Random color
        color = colors[i % len(colors)]
        
        # Create ball
        ball = library_call('ball', radius=radius, color=color)
        
        # Random position within a circular area
        angle = np.random.uniform(0, 2 * math.pi)
        distance = np.random.uniform(0, 0.2)
        x = distance * math.cos(angle)
        z = distance * math.sin(angle)
        y = radius + np.random.uniform(0, 0.05) * i  # Stack with some randomness
        
        return transform_shape(ball, translation_matrix([x, y, z]))
    
    return loop(num_balls, loop_fn)

@register()
def toy_train(length: float, color: tuple[float, float, float] = (0.7, 0.0, 0.0)) -> Shape:
    """Create a simple toy train with engine and cars"""
    # Engine body
    engine_body = primitive_call('cube', color=color, shape_kwargs={'scale': (0.15, 0.1, 0.2)})
    
    # Engine cabin
    engine_cabin = primitive_call('cube', color=(0.3, 0.3, 0.3), 
                                 shape_kwargs={'scale': (0.12, 0.08, 0.1)})
    engine_cabin = transform_shape(engine_cabin, translation_matrix([0, 0.09, -0.05]))
    
    # Wheels
    def create_wheel(x: float, z: float) -> Shape:
        wheel = primitive_call('cylinder', color=(0.2, 0.2, 0.2), 
                              shape_kwargs={'radius': 0.03, 'p0': (x, 0, z), 'p1': (x, 0.03, z)})
        return wheel
    
    wheels = concat_shapes(
        create_wheel(-0.05, -0.08),
        create_wheel(-0.05, 0.08),
        create_wheel(0.05, -0.08),
        create_wheel(0.05, 0.08)
    )
    
    # Smokestack
    smokestack = primitive_call('cylinder', color=(0.3, 0.3, 0.3), 
                               shape_kwargs={'radius': 0.02, 'p0': (0, 0.1, -0.07), 'p1': (0, 0.18, -0.07)})
    
    # Combine engine parts
    engine = concat_shapes(engine_body, engine_cabin, wheels, smokestack)
    
    # Create cars
    def create_car(position: float) -> Shape:
        car_color = (np.random.uniform(0.3, 0.9), np.random.uniform(0.3, 0.9), np.random.uniform(0.3, 0.9))
        car_body = primitive_call('cube', color=car_color, shape_kwargs={'scale': (0.12, 0.08, 0.15)})
        car_body = transform_shape(car_body, translation_matrix([0, 0, position]))
        
        car_wheels = concat_shapes(
            create_wheel(-0.04, position - 0.05),
            create_wheel(-0.04, position + 0.05),
            create_wheel(0.04, position - 0.05),
            create_wheel(0.04, position + 0.05)
        )
        
        return concat_shapes(car_body, car_wheels)
    
    # Create cars based on train length
    num_cars = max(1, int(length / 0.2) - 1)
    cars = concat_shapes(*[create_car(0.25 + i * 0.2) for i in range(num_cars)])
    
    return concat_shapes(engine, cars)

@register()
def play_mat(width: float, length: float, color: tuple[float, float, float] = (0.0, 0.7, 0.2)) -> Shape:
    """Create a play mat for the children's corner"""
    mat = primitive_call('cube', color=color, shape_kwargs={'scale': (width, 0.01, length)})
    return mat

@register()
def toy_shelf(width: float, height: float, depth: float) -> Shape:
    """Create a toy shelf with multiple compartments"""
    # Main shelf body
    shelf_body = primitive_call('cube', color=(0.8, 0.8, 0.8), 
                               shape_kwargs={'scale': (width, height, depth)})
    
    # Shelf dividers
    num_dividers = 2
    divider_width = 0.02
    
    dividers = []
    for i in range(1, num_dividers + 1):
        x_pos = (i / (num_dividers + 1) - 0.5) * width
        divider = primitive_call('cube', color=(0.75, 0.75, 0.75), 
                                shape_kwargs={'scale': (divider_width, height - 0.05, depth - 0.05)})
        divider = transform_shape(divider, translation_matrix([x_pos, 0, 0]))
        dividers.append(divider)
    
    # Horizontal shelf
    shelf = primitive_call('cube', color=(0.75, 0.75, 0.75), 
                          shape_kwargs={'scale': (width - 0.05, divider_width, depth - 0.05)})
    shelf = transform_shape(shelf, translation_matrix([0, height/4, 0]))
    
    return concat_shapes(shelf_body, *dividers, shelf)

@register()
def stuffed_animal(position: P, size: float, color: tuple[float, float, float]) -> Shape:
    """Create a simple stuffed animal (teddy bear)"""
    # Body
    body = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.15})
    body = transform_shape(body, translation_matrix([position[0], position[1], position[2]]))
    
    # Head
    head = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.1})
    head = transform_shape(head, translation_matrix([position[0], position[1] + size * 0.2, position[2]]))
    
    # Ears
    ear_left = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.04})
    ear_left = transform_shape(ear_left, translation_matrix([position[0] - size * 0.08, 
                                                           position[1] + size * 0.28, 
                                                           position[2]]))
    
    ear_right = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.04})
    ear_right = transform_shape(ear_right, translation_matrix([position[0] + size * 0.08, 
                                                             position[1] + size * 0.28, 
                                                             position[2]]))
    
    # Arms
    arm_left = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.06})
    arm_left = transform_shape(arm_left, translation_matrix([position[0] - size * 0.18, 
                                                           position[1] + size * 0.05, 
                                                           position[2]]))
    
    arm_right = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.06})
    arm_right = transform_shape(arm_right, translation_matrix([position[0] + size * 0.18, 
                                                             position[1] + size * 0.05, 
                                                             position[2]]))
    
    # Legs
    leg_left = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.07})
    leg_left = transform_shape(leg_left, translation_matrix([position[0] - size * 0.1, 
                                                           position[1] - size * 0.15, 
                                                           position[2]]))
    
    leg_right = primitive_call('sphere', color=color, shape_kwargs={'radius': size * 0.07})
    leg_right = transform_shape(leg_right, translation_matrix([position[0] + size * 0.1, 
                                                             position[1] - size * 0.15, 
                                                             position[2]]))
    
    # Nose
    nose = primitive_call('sphere', color=(0.1, 0.1, 0.1), shape_kwargs={'radius': size * 0.02})
    nose = transform_shape(nose, translation_matrix([position[0], 
                                                   position[1] + size * 0.2, 
                                                   position[2] + size * 0.09]))
    
    return concat_shapes(body, head, ear_left, ear_right, arm_left, arm_right, leg_left, leg_right, nose)

@register()
def children_playing_corner() -> Shape:
    """Create a complete children's playing corner with toys and furniture"""
    # Create play mat as the base
    mat = library_call('play_mat', width=2.0, length=2.0, color=(0.2, 0.8, 0.3))
    
    # Create toy shelf
    shelf = library_call('toy_shelf', width=0.8, height=0.6, depth=0.3)
    shelf = transform_shape(shelf, translation_matrix([0.5, 0.3, -0.7]))
    
    # Create blocks stack
    blocks = library_call('toy_blocks_stack', base_size=0.15, height=0.05, num_blocks=7)
    blocks = transform_shape(blocks, translation_matrix([-0.5, 0.025, -0.3]))
    
    # Create ball pile
    balls = library_call('toy_balls_pile', num_balls=12)
    balls = transform_shape(balls, translation_matrix([0.4, 0.025, 0.4]))
    
    # Create toy train
    train = library_call('toy_train', length=0.8)
    train = transform_shape(train, translation_matrix([-0.3, 0.03, 0.5]))
    train_center = compute_shape_center(train)
    train = transform_shape(train, rotation_matrix(math.pi/4, direction=(0, 1, 0), point=train_center))
    
    # Create stuffed animals
    teddy1 = library_call('stuffed_animal', position=[-0.6, 0.15, 0.0], size=0.8, color=(0.6, 0.4, 0.2))
    teddy2 = library_call('stuffed_animal', position=[0.0, 0.15, -0.5], size=0.7, color=(0.8, 0.7, 0.3))
    teddy2_center = compute_shape_center(teddy2)
    teddy2 = transform_shape(teddy2, rotation_matrix(math.pi/3, direction=(0, 1, 0), point=teddy2_center))
    
    # Add some random blocks on the shelf
    shelf_blocks = library_call('toy_blocks_stack', base_size=0.1, height=0.05, num_blocks=3)
    shelf_blocks = transform_shape(shelf_blocks, translation_matrix([0.3, 0.6, -0.7]))
    
    # Add a ball on the shelf
    shelf_ball = library_call('ball', radius=0.07, color=(0.0, 0.5, 1.0))
    shelf_ball = transform_shape(shelf_ball, translation_matrix([0.7, 0.65, -0.7]))
    
    return concat_shapes(
        mat, shelf, blocks, balls, train, teddy1, teddy2, shelf_blocks, shelf_ball
    )
```