```python
from helper import *

"""
Sponge Bob and friends
"""

@register()
def sponge_bob() -> Shape:
    # Create SpongeBob's body (yellow sponge) - more square shape
    body = primitive_call('cube', shape_kwargs={'scale': (0.8, 1.0, 0.4)}, color=(1.0, 0.9, 0.0))

    # Create SpongeBob's pants
    pants = primitive_call('cube', shape_kwargs={'scale': (0.8, 0.4, 0.4)}, color=(0.6, 0.6, 1.0))
    pants = transform_shape(pants, translation_matrix((0, -0.7, 0)))

    # Create SpongeBob's belt
    belt = primitive_call('cube', shape_kwargs={'scale': (0.8, 0.05, 0.41)}, color=(0.0, 0.0, 0.0))
    belt = transform_shape(belt, translation_matrix((0, -0.5, 0)))

    # Create SpongeBob's sponge holes
    holes = []
    np.random.seed(42)
    for _ in range(15):
        x = np.random.uniform(-0.3, 0.3)
        y = np.random.uniform(-0.4, 0.4)
        z = np.random.uniform(-0.2, 0.2)

        hole = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.9, 0.8, 0.0))
        hole = transform_shape(hole, translation_matrix((x, y, z)))
        holes.append(hole)

    # Create SpongeBob's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.2, 0.3, -0.21)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.2, 0.3, -0.21)))

    # Create SpongeBob's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.2, 0.3, -0.28)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.2, 0.3, -0.28)))

    # Create SpongeBob's nose
    nose = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 0.8, 0.0))
    nose = transform_shape(nose, translation_matrix((0.0, 0.0, -0.25)))

    # Create SpongeBob's mouth
    mouth = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.3, -0.2, -0.21), 'p1': (0.3, -0.2, -0.21)}, color=(0.8, 0.0, 0.0))

    # Create SpongeBob's arms
    left_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.4, 0.0, 0.0), 'p1': (-0.8, -0.2, 0.0)}, color=(1.0, 0.9, 0.0))
    right_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.4, 0.0, 0.0), 'p1': (0.8, -0.2, 0.0)}, color=(1.0, 0.9, 0.0))

    # Create SpongeBob's legs
    left_leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.25, -0.9, 0.0), 'p1': (-0.25, -1.4, 0.0)}, color=(1.0, 0.9, 0.0))
    right_leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.25, -0.9, 0.0), 'p1': (0.25, -1.4, 0.0)}, color=(1.0, 0.9, 0.0))

    # Create SpongeBob's shoes
    left_shoe = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.0, 0.0, 0.0))
    left_shoe = transform_shape(left_shoe, translation_matrix((-0.25, -1.4, 0.0)))

    right_shoe = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.0, 0.0, 0.0))
    right_shoe = transform_shape(right_shoe, translation_matrix((0.25, -1.4, 0.0)))

    return concat_shapes(body, pants, belt, *holes, left_eye, right_eye, left_pupil, right_pupil, nose, mouth,
                         left_arm, right_arm, left_leg, right_leg, left_shoe, right_shoe)

@register()
def patrick() -> Shape:
    # Create Patrick's body with a more distinct star shape
    body_center = primitive_call('sphere', shape_kwargs={'radius': 0.3}, color=(1.0, 0.6, 0.6))
    
    # Create star points for Patrick's body - make them more pronounced
    star_points = []
    for i in range(5):
        angle = 2 * math.pi * i / 5 + math.pi/10  # Offset to make it stand on two legs
        x = 0.6 * math.cos(angle)
        y = 0.6 * math.sin(angle)
        z = 0
        
        # Create a cone-like shape for each star point
        point = primitive_call('cylinder', shape_kwargs={'radius': 0.25, 'p0': (0, 0, 0), 'p1': (x, y, z)}, color=(1.0, 0.6, 0.6))
        star_points.append(point)
        
        # Add a sphere at the tip of each point to make it more rounded
        tip = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 0.6, 0.6))
        tip = transform_shape(tip, translation_matrix((x, y, z)))
        star_points.append(tip)

    # Create Patrick's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.2, -0.35)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.2, -0.35)))

    # Create Patrick's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.2, -0.4)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.2, -0.4)))

    # Create Patrick's mouth
    mouth = primitive_call('cylinder', shape_kwargs={'radius': 0.04, 'p0': (-0.2, -0.1, -0.35), 'p1': (0.2, -0.1, -0.35)}, color=(0.8, 0.0, 0.0))

    # Create Patrick's shorts
    shorts = primitive_call('cube', shape_kwargs={'scale': (0.6, 0.3, 0.4)}, color=(0.5, 0.0, 0.5))
    shorts = transform_shape(shorts, translation_matrix((0, -0.4, 0)))

    return concat_shapes(body_center, *star_points, left_eye, right_eye, left_pupil, right_pupil, mouth, shorts)

@register()
def squidward() -> Shape:
    # Create Squidward's head (turquoise) - more elongated
    head = primitive_call('sphere', shape_kwargs={'radius': 0.4}, color=(0.0, 0.7, 0.7))
    head = transform_shape(head, scale_matrix(0.8, (0, 0, 0)))
    head = transform_shape(head, translation_matrix((0, 0.1, 0)))  # Move head up slightly

    # Create Squidward's elongated forehead
    forehead = primitive_call('sphere', shape_kwargs={'radius': 0.3}, color=(0.0, 0.7, 0.7))
    forehead = transform_shape(forehead, scale_matrix(0.8, (0, 0, 0)))
    forehead = transform_shape(forehead, translation_matrix((0, 0.4, 0)))

    # Create Squidward's nose
    nose = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0, 0, -0.3), 'p1': (0, 0, -0.7)}, color=(0.0, 0.7, 0.7))

    # Create Squidward's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.15, -0.3)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.15, -0.3)))

    # Create Squidward's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.15, -0.37)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.15, -0.37)))

    # Create Squidward's body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.25, 'p0': (0, -0.3, 0), 'p1': (0, -1.0, 0)}, color=(0.0, 0.7, 0.7))

    # Create Squidward's shirt
    shirt = primitive_call('cylinder', shape_kwargs={'radius': 0.26, 'p0': (0, -0.3, 0), 'p1': (0, -0.8, 0)}, color=(0.6, 0.4, 0.1))

    # Create Squidward's tentacles with more natural curve
    tentacles = []
    for i in range(6):
        angle = math.pi + (math.pi * i / 5)
        x = 0.3 * math.cos(angle)
        z = 0.3 * math.sin(angle)

        # Create curved tentacles with multiple segments for more natural look
        segments = []
        prev_x, prev_y, prev_z = x, -1.0, z

        # Create more segments with smoother curves
        for j in range(5):  # Increased from 3 to 5 segments
            curve_factor = math.sin(j * 0.5) * 0.15
            next_x = x + curve_factor + 0.05*j*math.cos(angle)
            next_y = -1.0 - j*0.15
            next_z = z + curve_factor + 0.05*j*math.sin(angle)

            segment = primitive_call('cylinder', shape_kwargs={'radius': 0.05,
                                                             'p0': (prev_x, prev_y, prev_z),
                                                             'p1': (next_x, next_y, next_z)},
                                   color=(0.0, 0.7, 0.7))
            segments.append(segment)
            
            # Add small sphere at each joint for smoother appearance
            if j > 0:
                joint = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(0.0, 0.7, 0.7))
                joint = transform_shape(joint, translation_matrix((prev_x, prev_y, prev_z)))
                segments.append(joint)
                
            prev_x, prev_y, prev_z = next_x, next_y, next_z

        tentacles.extend(segments)

    return concat_shapes(head, forehead, nose, left_eye, right_eye, left_pupil, right_pupil, body, shirt, *tentacles)

@register()
def mr_krabs() -> Shape:
    # Create Mr. Krabs' body (red)
    body = primitive_call('sphere', shape_kwargs={'radius': 0.4}, color=(0.9, 0.2, 0.1))

    # Create Mr. Krabs' eyes (on stalks)
    left_stalk = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.1, 0.2, 0), 'p1': (-0.1, 0.5, -0.2)}, color=(0.9, 0.2, 0.1))
    right_stalk = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.1, 0.2, 0), 'p1': (0.1, 0.5, -0.2)}, color=(0.9, 0.2, 0.1))

    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.1, 0.5, -0.2)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.1, 0.5, -0.2)))

    # Create Mr. Krabs' pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.1, 0.5, -0.27)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.1, 0.5, -0.27)))

    # Create Mr. Krabs' claws (more prominent)
    # Left claw
    left_claw_base = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.9, 0.2, 0.1))
    left_claw_base = transform_shape(left_claw_base, translation_matrix((-0.6, 0.0, 0.0)))

    left_claw_upper = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (-0.6, 0.0, 0.0), 'p1': (-0.9, 0.15, -0.1)}, color=(0.9, 0.2, 0.1))
    left_claw_lower = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (-0.6, 0.0, 0.0), 'p1': (-0.9, -0.15, -0.1)}, color=(0.9, 0.2, 0.1))

    # Right claw
    right_claw_base = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.9, 0.2, 0.1))
    right_claw_base = transform_shape(right_claw_base, translation_matrix((0.6, 0.0, 0.0)))

    right_claw_upper = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0.6, 0.0, 0.0), 'p1': (0.9, 0.15, -0.1)}, color=(0.9, 0.2, 0.1))
    right_claw_lower = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0.6, 0.0, 0.0), 'p1': (0.9, -0.15, -0.1)}, color=(0.9, 0.2, 0.1))

    # Create Mr. Krabs' legs
    legs = []
    for i in range(4):
        x_offset = 0.2 if i % 2 == 0 else -0.2
        z_offset = 0.2 if i < 2 else -0.2

        leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x_offset, -0.2, z_offset), 'p1': (x_offset*1.5, -0.8, z_offset*1.5)}, color=(0.9, 0.2, 0.1))
        legs.append(leg)

    # Create Mr. Krabs' shirt
    shirt = primitive_call('cylinder', shape_kwargs={'radius': 0.41, 'p0': (0, -0.2, 0), 'p1': (0, -0.6, 0)}, color=(0.1, 0.5, 0.9))

    return concat_shapes(body, left_stalk, right_stalk, left_eye, right_eye,
                         left_pupil, right_pupil, left_claw_base, left_claw_upper, left_claw_lower,
                         right_claw_base, right_claw_upper, right_claw_lower, shirt, *legs)

@register()
def sandy() -> Shape:
    # Create Sandy's helmet (more transparent)
    helmet = primitive_call('sphere', shape_kwargs={'radius': 0.45}, color=(0.9, 0.9, 1.0))  # More transparent color

    # Create helmet collar
    collar = primitive_call('cylinder', shape_kwargs={'radius': 0.45, 'p0': (0, -0.45, 0), 'p1': (0, -0.55, 0)}, color=(0.7, 0.7, 0.7))

    # Create air tube
    air_tube = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, -0.5, 0.4), 'p1': (0, -0.5, 0.6)}, color=(0.7, 0.7, 0.7))

    # Create Sandy's head (squirrel) - more squirrel-like
    head = primitive_call('sphere', shape_kwargs={'radius': 0.3}, color=(0.8, 0.6, 0.4))

    # Create Sandy's ears - more pronounced
    left_ear = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(0.8, 0.6, 0.4))
    left_ear = transform_shape(left_ear, scale_matrix(1.0, (0, 0, 0)))
    left_ear = transform_shape(left_ear, translation_matrix((-0.2, 0.35, 0)))

    right_ear = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(0.8, 0.6, 0.4))
    right_ear = transform_shape(right_ear, scale_matrix(1.0, (0, 0, 0)))
    right_ear = transform_shape(right_ear, translation_matrix((0.2, 0.35, 0)))

    # Create Sandy's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.08}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.1, -0.25)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.08}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.1, -0.25)))

    # Create Sandy's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.1, -0.3)))

    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.1, -0.3)))

    # Create Sandy's nose
    nose = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.6, 0.3, 0.3))
    nose = transform_shape(nose, translation_matrix((0.0, 0.0, -0.3)))

    # Create Sandy's body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.25, 'p0': (0, -0.45, 0), 'p1': (0, -1.0, 0)}, color=(1.0, 1.0, 1.0))

    # Create Sandy's suit
    suit = primitive_call('cylinder', shape_kwargs={'radius': 0.26, 'p0': (0, -0.45, 0), 'p1': (0, -0.8, 0)}, color=(0.0, 0.3, 0.8))

    # Create Sandy's arms
    left_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (-0.25, -0.6, 0), 'p1': (-0.5, -0.4, 0)}, color=(1.0, 1.0, 1.0))
    right_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0.25, -0.6, 0), 'p1': (0.5, -0.4, 0)}, color=(1.0, 1.0, 1.0))

    # Create Sandy's tail - more bushy
    tail_base = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.8, 0.6, 0.4))
    tail_base = transform_shape(tail_base, translation_matrix((0.0, -0.9, 0.3)))
    
    tail_tip = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.8, 0.6, 0.4))
    tail_tip = transform_shape(tail_tip, translation_matrix((0.0, -0.8, 0.5)))

    return concat_shapes(helmet, collar, air_tube, head, left_ear, right_ear, left_eye, right_eye, left_pupil, right_pupil,
                         nose, body, suit, left_arm, right_arm, tail_base, tail_tip)

@register()
def create_seaweed(x: float, z: float, height: float, floor_y: float) -> Shape:
    segments = []
    prev_x, prev_y, prev_z = x, floor_y, z  # Start from floor surface

    # Create curved seaweed with multiple segments - thicker and more visible
    for i in range(8):  # More segments for smoother curve
        segment_height = height / 8
        wave_factor = 0.15 * math.sin(i * 0.8)  # More pronounced wave
        next_x = x + wave_factor
        next_y = prev_y + segment_height
        next_z = z + wave_factor * math.cos(i * 0.8)

        segment = primitive_call('cylinder', shape_kwargs={'radius': 0.05,  # Thicker seaweed
                                                         'p0': (prev_x, prev_y, prev_z),
                                                         'p1': (next_x, next_y, next_z)},
                               color=(0.0, 0.6, 0.3))
        segments.append(segment)
        prev_x, prev_y, prev_z = next_x, next_y, next_z

    return concat_shapes(*segments)

@register()
def create_bubble(x: float, y: float, z: float, size: float) -> Shape:
    # Create a bubble with slight transparency
    bubble = primitive_call('sphere', shape_kwargs={'radius': size}, color=(0.8, 0.9, 1.0))
    bubble = transform_shape(bubble, translation_matrix((x, y, z)))
    return bubble

@register()
def create_coral(x: float, z: float, floor_y: float) -> Shape:
    # Create a coral formation
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.1, 
                                                  'p0': (x, floor_y, z), 
                                                  'p1': (x, floor_y + 0.3, z)}, 
                        color=(0.9, 0.5, 0.5))
    
    branches = []
    np.random.seed(int(x*100 + z*100))  # Unique seed for each coral
    
    for i in range(3):
        angle = np.random.uniform(0, 2*math.pi)
        branch_x = x + 0.1 * math.cos(angle)
        branch_z = z + 0.1 * math.sin(angle)
        height = np.random.uniform(0.2, 0.4)
        
        branch = primitive_call('cylinder', shape_kwargs={'radius': 0.05,
                                                        'p0': (x, floor_y + 0.2, z),
                                                        'p1': (branch_x, floor_y + 0.2 + height, branch_z)},
                              color=(1.0, 0.4, 0.4))
        branches.append(branch)
        
        # Add a small sphere at the tip of each branch
        tip = primitive_call('sphere', shape_kwargs={'radius': 0.07}, 
                           color=(1.0, 0.3, 0.3))
        tip = transform_shape(tip, translation_matrix((branch_x, floor_y + 0.2 + height, branch_z)))
        branches.append(tip)
    
    return concat_shapes(base, *branches)

@register()
def pineapple_house() -> Shape:
    # Create the pineapple body
    body = primitive_call('sphere', shape_kwargs={'radius': 0.8}, color=(1.0, 0.8, 0.0))
    body = transform_shape(body, scale_matrix(1.5, (0, 0, 0)))

    # Create the pineapple texture (small bumps)
    bumps = []
    np.random.seed(43)
    for _ in range(50):
        angle = np.random.uniform(0, 2*math.pi)
        height = np.random.uniform(-0.8, 0.8)
        radius = 0.8

        x = radius * math.cos(angle)
        z = radius * math.sin(angle)

        bump = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(0.9, 0.7, 0.0))
        bump = transform_shape(bump, translation_matrix((x, height, z)))
        bumps.append(bump)

    # Create the pineapple leaves
    leaves = []
    for i in range(8):
        angle = 2 * math.pi * i / 8
        x = 0.3 * math.cos(angle)
        z = 0.3 * math.sin(angle)

        leaf = primitive_call('cylinder', shape_kwargs={'radius': 0.05,
                                                      'p0': (0, 1.2, 0),
                                                      'p1': (x, 1.6, z)},
                            color=(0.0, 0.5, 0.0))
        leaves.append(leaf)

    # Create the door
    door = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.6, 0.1)}, color=(0.6, 0.3, 0.1))
    door = transform_shape(door, translation_matrix((0, -0.6, -0.8)))

    # Create windows
    window1 = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.7, 0.9, 1.0))
    window1 = transform_shape(window1, translation_matrix((0.5, 0.3, -0.6)))

    window2 = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.7, 0.9, 1.0))
    window2 = transform_shape(window2, translation_matrix((-0.5, 0.3, -0.6)))

    return concat_shapes(body, *bumps, *leaves, door, window1, window2)

@register()
def squidward_house() -> Shape:
    # Create the Easter Island head body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.8, 'p0': (0, -1.0, 0), 'p1': (0, 1.5, 0)}, color=(0.5, 0.5, 0.5))

    # Create the head top
    top = primitive_call('sphere', shape_kwargs={'radius': 0.8}, color=(0.5, 0.5, 0.5))
    top = transform_shape(top, translation_matrix((0, 1.5, 0)))

    # Create the nose
    nose = primitive_call('cylinder', shape_kwargs={'radius': 0.2, 'p0': (0, 0.5, 0), 'p1': (0, 0.5, -1.0)}, color=(0.5, 0.5, 0.5))

    # Create the eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.7, 0.9, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.4, 1.0, -0.7)))

    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.7, 0.9, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.4, 1.0, -0.7)))

    # Create the door
    door = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.8, 0.1)}, color=(0.6, 0.3, 0.1))
    door = transform_shape(door, translation_matrix((0, -0.6, -0.8)))

    return concat_shapes(body, top, nose, left_eye, right_eye, door)

@register()
def patrick_rock() -> Shape:
    # Create the rock
    rock = primitive_call('sphere', shape_kwargs={'radius': 0.8}, color=(0.6, 0.6, 0.6))
    rock = transform_shape(rock, scale_matrix(1.0, (0, 0, 0)))
    rock = transform_shape(rock, translation_matrix((0, -0.4, 0)))  # Partially buried

    # Create some texture/details
    details = []
    np.random.seed(44)
    for _ in range(10):
        x = np.random.uniform(-0.6, 0.6)
        y = np.random.uniform(-0.6, 0.2)
        z = np.random.uniform(-0.6, 0.6)

        detail = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(0.5, 0.5, 0.5))
        detail = transform_shape(detail, translation_matrix((x, y, z)))
        details.append(detail)

    return concat_shapes(rock, *details)

@register()
def krusty_krab() -> Shape:
    # Create the main building (a crab trap shape)
    main_body = primitive_call('cylinder', shape_kwargs={'radius': 1.0, 'p0': (0, -0.5, 0), 'p1': (0, 0.5, 0)}, color=(0.8, 0.4, 0.1))
    
    # Create the roof (dome-shaped)
    roof = primitive_call('sphere', shape_kwargs={'radius': 1.0}, color=(0.8, 0.4, 0.1))
    roof = transform_shape(roof, scale_matrix(1.0, (0, 0, 0)))
    roof = transform_shape(roof, translation_matrix((0, 0.5, 0)))
    
    # Create the door
    door = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.6, 0.1)}, color=(0.6, 0.3, 0.1))
    door = transform_shape(door, translation_matrix((0, -0.2, -1.0)))
    
    # Create the sign
    sign_post = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0.5, -1.0), 'p1': (0, 1.2, -1.0)}, color=(0.6, 0.3, 0.1))
    
    sign_board = primitive_call('cube', shape_kwargs={'scale': (0.8, 0.3, 0.1)}, color=(0.9, 0.1, 0.1))
    sign_board = transform_shape(sign_board, translation_matrix((0, 1.2, -1.0)))
    
    # Create windows
    window1 = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.7, 0.9, 1.0))
    window1 = transform_shape(window1, translation_matrix((0.6, 0.0, -0.8)))
    
    window2 = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.7, 0.9, 1.0))
    window2 = transform_shape(window2, translation_matrix((-0.6, 0.0, -0.8)))
    
    return concat_shapes(main_body, roof, door, sign_post, sign_board, window1, window2)

@register()
def bikini_bottom_scene() -> Shape:
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create the ocean floor
    floor = primitive_call('cube', shape_kwargs={'scale': (15, 0.1, 15)}, color=(0.8, 0.7, 0.2))
    floor = transform_shape(floor, translation_matrix((0, -1.5, 0)))

    # Calculate floor top surface
    floor_top_y = -1.5 + 0.05  # Half the floor height

    # Create the characters
    spongebob = library_call('sponge_bob')
    patrick = library_call('patrick')
    squidward = library_call('squidward')
    mr_krabs = library_call('mr_krabs')
    sandy = library_call('sandy')

    # Create houses
    pineapple = library_call('pineapple_house')
    easter_head = library_call('squidward_house')
    rock = library_call('patrick_rock')
    krusty_krab = library_call('krusty_krab')

    # Calculate character heights to position them on the floor
    spongebob_min_y = compute_shape_min(spongebob)[1]
    patrick_min_y = compute_shape_min(patrick)[1]
    squidward_min_y = compute_shape_min(squidward)[1]
    mr_krabs_min_y = compute_shape_min(mr_krabs)[1]
    sandy_min_y = compute_shape_min(sandy)[1]

    # Calculate house heights
    pineapple_min_y = compute_shape_min(pineapple)[1]
    easter_head_min_y = compute_shape_min(easter_head)[1]
    rock_min_y = compute_shape_min(rock)[1]
    krusty_krab_min_y = compute_shape_min(krusty_krab)[1]

    # Scale characters to be proportionally larger relative to houses
    character_scale = 1.2
    spongebob = transform_shape(spongebob, scale_matrix(character_scale, (0, 0, 0)))
    patrick = transform_shape(patrick, scale_matrix(character_scale, (0, 0, 0)))
    squidward = transform_shape(squidward, scale_matrix(character_scale, (0, 0, 0)))
    mr_krabs = transform_shape(mr_krabs, scale_matrix(character_scale, (0, 0, 0)))
    sandy = transform_shape(sandy, scale_matrix(character_scale, (0, 0, 0)))

    # Recalculate min_y after scaling
    spongebob_min_y = compute_shape_min(spongebob)[1]
    patrick_min_y = compute_shape_min(patrick)[1]
    squidward_min_y = compute_shape_min(squidward)[1]
    mr_krabs_min_y = compute_shape_min(mr_krabs)[1]
    sandy_min_y = compute_shape_min(sandy)[1]

    # Position houses
    pineapple = transform_shape(pineapple, translation_matrix((2.5, floor_top_y - pineapple_min_y, -2.0)))
    easter_head = transform_shape(easter_head, translation_matrix((-2.5, floor_top_y - easter_head_min_y, -2.0)))
    rock = transform_shape(rock, translation_matrix((1.5, floor_top_y - rock_min_y, 2.0)))
    krusty_krab = transform_shape(krusty_krab, translation_matrix((-3.0, floor_top_y - krusty_krab_min_y, 3.0)))

    # Position the characters near their respective houses
    spongebob = transform_shape(spongebob, translation_matrix((2.0, floor_top_y - spongebob_min_y, -1.5)))  # Near pineapple
    patrick = transform_shape(patrick, translation_matrix((1.2, floor_top_y - patrick_min_y, 1.5)))  # Near rock
    squidward = transform_shape(squidward, translation_matrix((-2.0, floor_top_y - squidward_min_y, -1.5)))  # Near Easter Island head
    mr_krabs = transform_shape(mr_krabs, translation_matrix((-2.5, floor_top_y - mr_krabs_min_y, 2.5)))  # Near Krusty Krab
    sandy = transform_shape(sandy, translation_matrix((0.0, floor_top_y - sandy_min_y, 0.0)))  # Center of scene

    # Create underwater environment elements
    
    # Create more and thicker seaweed
    seaweeds = []
    for i in range(20):  # Increased from 8 to 20
        x = np.random.uniform(-6, 6)
        z = np.random.uniform(-6, 6)
        height = np.random.uniform(0.8, 2.0)  # Taller seaweed

        seaweed = library_call('create_seaweed', x=x, z=z, height=height, floor_y=floor_top_y)
        seaweeds.append(seaweed)

    # Create coral formations
    corals = []
    for i in range(10):
        x = np.random.uniform(-6, 6)
        z = np.random.uniform(-6, 6)
        
        coral = library_call('create_coral', x=x, z=z, floor_y=floor_top_y)
        corals.append(coral)

    # Create bubbles floating upward
    bubbles = []
    for i in range(15):
        x = np.random.uniform(-6, 6)
        y = np.random.uniform(-1, 3)
        z = np.random.uniform(-6, 6)
        size = np.random.uniform(0.05, 0.15)
        
        bubble = library_call('create_bubble', x=x, y=y, z=z, size=size)
        bubbles.append(bubble)

    # Create some rocks
    rocks = []
    for i in range(10):  # Increased from 5 to 10
        x = np.random.uniform(-6, 6)
        z = np.random.uniform(-6, 6)
        size = np.random.uniform(0.2, 0.5)

        rock = primitive_call('sphere', shape_kwargs={'radius': size}, color=(0.5, 0.5, 0.5))
        rock = transform_shape(rock, translation_matrix((x, floor_top_y + size/2, z)))
        rocks.append(rock)

    # Create blue water tint/fog
    water_tint = primitive_call('cube', shape_kwargs={'scale': (15, 6, 15)}, color=(0.6, 0.8, 1.0))
    water_tint = transform_shape(water_tint, translation_matrix((0, 1.5, 0)))

    return concat_shapes(spongebob, patrick, squidward, mr_krabs, sandy,
                         pineapple, easter_head, rock, krusty_krab, floor, 
                         *seaweeds, *corals, *bubbles, *rocks)
```