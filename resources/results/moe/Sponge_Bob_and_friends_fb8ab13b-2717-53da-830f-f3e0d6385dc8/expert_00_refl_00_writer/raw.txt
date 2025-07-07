```python
from helper import *

"""
Sponge Bob and friends
"""

@register()
def sponge_bob() -> Shape:
    # Create SpongeBob's body (yellow sponge)
    body = primitive_call('cube', shape_kwargs={'scale': (0.8, 1.0, 0.4)}, color=(1.0, 0.9, 0.0))
    
    # Create SpongeBob's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.2, 0.3, -0.21)))
    
    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.2, 0.3, -0.21)))
    
    # Create SpongeBob's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.2, 0.3, -0.35)))
    
    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.2, 0.3, -0.35)))
    
    # Create SpongeBob's nose
    nose = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 0.8, 0.0))
    nose = transform_shape(nose, translation_matrix((0.0, 0.0, -0.25)))
    
    # Create SpongeBob's mouth
    mouth = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.3, -0.2, -0.21), 'p1': (0.3, -0.2, -0.21)}, color=(0.8, 0.0, 0.0))
    
    # Create SpongeBob's arms
    left_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.4, 0.0, 0.0), 'p1': (-0.8, -0.2, 0.0)}, color=(1.0, 0.9, 0.0))
    right_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.4, 0.0, 0.0), 'p1': (0.8, -0.2, 0.0)}, color=(1.0, 0.9, 0.0))
    
    # Create SpongeBob's legs
    left_leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (-0.25, -0.5, 0.0), 'p1': (-0.25, -1.0, 0.0)}, color=(1.0, 0.9, 0.0))
    right_leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0.25, -0.5, 0.0), 'p1': (0.25, -1.0, 0.0)}, color=(1.0, 0.9, 0.0))
    
    # Create SpongeBob's shoes
    left_shoe = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.0, 0.0, 0.0))
    left_shoe = transform_shape(left_shoe, translation_matrix((-0.25, -1.0, 0.0)))
    
    right_shoe = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.0, 0.0, 0.0))
    right_shoe = transform_shape(right_shoe, translation_matrix((0.25, -1.0, 0.0)))
    
    return concat_shapes(body, left_eye, right_eye, left_pupil, right_pupil, nose, mouth, 
                         left_arm, right_arm, left_leg, right_leg, left_shoe, right_shoe)

@register()
def patrick() -> Shape:
    # Create Patrick's body (pink starfish)
    body = primitive_call('sphere', shape_kwargs={'radius': 0.5}, color=(1.0, 0.6, 0.6))
    body = transform_shape(body, scale_matrix(1.2, (0, 0, 0)))
    
    # Create Patrick's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.2, -0.45)))
    
    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.2, -0.45)))
    
    # Create Patrick's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.2, -0.55)))
    
    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.2, -0.55)))
    
    # Create Patrick's mouth
    mouth = primitive_call('cylinder', shape_kwargs={'radius': 0.04, 'p0': (-0.2, -0.1, -0.45), 'p1': (0.2, -0.1, -0.45)}, color=(0.8, 0.0, 0.0))
    
    # Create Patrick's arms and legs (starfish points)
    limbs = []
    
    # Create 5 limbs in a star pattern
    for i in range(5):
        angle = 2 * math.pi * i / 5 + math.pi/10  # Offset to make it stand on two legs
        x = 0.7 * math.cos(angle)
        y = 0.7 * math.sin(angle)
        
        limb = primitive_call('cylinder', shape_kwargs={'radius': 0.1, 'p0': (0, 0, 0), 'p1': (x, y, 0)}, color=(1.0, 0.6, 0.6))
        limbs.append(limb)
    
    return concat_shapes(body, left_eye, right_eye, left_pupil, right_pupil, mouth, *limbs)

@register()
def squidward() -> Shape:
    # Create Squidward's head (turquoise)
    head = primitive_call('sphere', shape_kwargs={'radius': 0.4}, color=(0.0, 0.7, 0.7))
    head = transform_shape(head, scale_matrix(0.8, (0, 0, 0)))
    
    # Create Squidward's nose
    nose = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0, 0, -0.3), 'p1': (0, 0, -0.7)}, color=(0.0, 0.7, 0.7))
    
    # Create Squidward's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.15, -0.3)))
    
    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.12}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.15, -0.3)))
    
    # Create Squidward's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.15, -0.42)))
    
    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.15, -0.42)))
    
    # Create Squidward's body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.25, 'p0': (0, -0.3, 0), 'p1': (0, -1.0, 0)}, color=(0.0, 0.7, 0.7))
    
    # Create Squidward's tentacles
    tentacles = []
    for i in range(6):
        angle = math.pi + (math.pi * i / 5)
        x = 0.3 * math.cos(angle)
        z = 0.3 * math.sin(angle)
        
        tentacle = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x, -1.0, z), 'p1': (x*1.5, -1.5, z*1.5)}, color=(0.0, 0.7, 0.7))
        tentacles.append(tentacle)
    
    return concat_shapes(head, nose, left_eye, right_eye, left_pupil, right_pupil, body, *tentacles)

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
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.1, 0.5, -0.3)))
    
    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.04}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.1, 0.5, -0.3)))
    
    # Create Mr. Krabs' claws
    left_claw = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.9, 0.2, 0.1))
    left_claw = transform_shape(left_claw, translation_matrix((-0.6, 0.0, 0.0)))
    left_claw = transform_shape(left_claw, scale_matrix(1.2, (-0.6, 0.0, 0.0)))
    
    right_claw = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.9, 0.2, 0.1))
    right_claw = transform_shape(right_claw, translation_matrix((0.6, 0.0, 0.0)))
    right_claw = transform_shape(right_claw, scale_matrix(1.2, (0.6, 0.0, 0.0)))
    
    # Create Mr. Krabs' legs
    legs = []
    for i in range(4):
        x_offset = 0.2 if i % 2 == 0 else -0.2
        z_offset = 0.2 if i < 2 else -0.2
        
        leg = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x_offset, -0.2, z_offset), 'p1': (x_offset*1.5, -0.8, z_offset*1.5)}, color=(0.9, 0.2, 0.1))
        legs.append(leg)
    
    return concat_shapes(body, left_stalk, right_stalk, left_eye, right_eye, 
                         left_pupil, right_pupil, left_claw, right_claw, *legs)

@register()
def sandy() -> Shape:
    # Create Sandy's helmet (transparent sphere)
    helmet = primitive_call('sphere', shape_kwargs={'radius': 0.5}, color=(0.8, 0.8, 1.0))
    
    # Create Sandy's head (squirrel)
    head = primitive_call('sphere', shape_kwargs={'radius': 0.35}, color=(0.8, 0.6, 0.4))
    
    # Create Sandy's eyes
    left_eye = primitive_call('sphere', shape_kwargs={'radius': 0.08}, color=(1.0, 1.0, 1.0))
    left_eye = transform_shape(left_eye, translation_matrix((-0.15, 0.1, -0.3)))
    
    right_eye = primitive_call('sphere', shape_kwargs={'radius': 0.08}, color=(1.0, 1.0, 1.0))
    right_eye = transform_shape(right_eye, translation_matrix((0.15, 0.1, -0.3)))
    
    # Create Sandy's pupils
    left_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 0.0, 0.0))
    left_pupil = transform_shape(left_pupil, translation_matrix((-0.15, 0.1, -0.38)))
    
    right_pupil = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 0.0, 0.0))
    right_pupil = transform_shape(right_pupil, translation_matrix((0.15, 0.1, -0.38)))
    
    # Create Sandy's nose
    nose = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.6, 0.3, 0.3))
    nose = transform_shape(nose, translation_matrix((0.0, 0.0, -0.35)))
    
    # Create Sandy's body
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.3, 'p0': (0, -0.5, 0), 'p1': (0, -1.2, 0)}, color=(1.0, 1.0, 1.0))
    
    # Create Sandy's arms
    left_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (-0.3, -0.6, 0), 'p1': (-0.6, -0.4, 0)}, color=(1.0, 1.0, 1.0))
    right_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0.3, -0.6, 0), 'p1': (0.6, -0.4, 0)}, color=(1.0, 1.0, 1.0))
    
    # Create Sandy's tail
    tail = primitive_call('sphere', shape_kwargs={'radius': 0.2}, color=(0.8, 0.6, 0.4))
    tail = transform_shape(tail, translation_matrix((0.0, -1.0, 0.4)))
    tail = transform_shape(tail, scale_matrix(1.5, (0.0, -1.0, 0.4)))
    
    return concat_shapes(helmet, head, left_eye, right_eye, left_pupil, right_pupil, 
                         nose, body, left_arm, right_arm, tail)

@register()
def bikini_bottom_scene() -> Shape:
    # Create the characters
    spongebob = library_call('sponge_bob')
    patrick = library_call('patrick')
    squidward = library_call('squidward')
    mr_krabs = library_call('mr_krabs')
    sandy = library_call('sandy')
    
    # Position the characters
    spongebob = transform_shape(spongebob, translation_matrix((0, 0, 0)))
    patrick = transform_shape(patrick, translation_matrix((1.5, -0.2, 0.5)))
    squidward = transform_shape(squidward, translation_matrix((-1.5, 0.2, 0.3)))
    mr_krabs = transform_shape(mr_krabs, translation_matrix((0.8, 0, -1.5)))
    sandy = transform_shape(sandy, translation_matrix((-0.8, 0, -1.2)))
    
    # Create the ocean floor
    floor = primitive_call('cube', shape_kwargs={'scale': (10, 0.1, 10)}, color=(0.8, 0.7, 0.2))
    floor = transform_shape(floor, translation_matrix((0, -1.5, 0)))
    
    # Create some seaweed
    seaweeds = []
    for i in range(8):
        x = np.random.uniform(-4, 4)
        z = np.random.uniform(-4, 4)
        height = np.random.uniform(0.5, 1.5)
        
        seaweed = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (x, -1.45, z), 'p1': (x, -1.45 + height, z)}, color=(0.0, 0.6, 0.3))
        seaweeds.append(seaweed)
    
    # Create some rocks
    rocks = []
    for i in range(5):
        x = np.random.uniform(-4, 4)
        z = np.random.uniform(-4, 4)
        size = np.random.uniform(0.2, 0.5)
        
        rock = primitive_call('sphere', shape_kwargs={'radius': size}, color=(0.5, 0.5, 0.5))
        rock = transform_shape(rock, translation_matrix((x, -1.45 + size/2, z)))
        rocks.append(rock)
    
    return concat_shapes(spongebob, patrick, squidward, mr_krabs, sandy, floor, *seaweeds, *rocks)
```