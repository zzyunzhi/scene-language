```python
from helper import *

"""
Basílica de la Sagrada Família
"""

@register()
def tower(height: float, base_radius: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a tower with a conical top"""
    base_height = height * 0.7
    top_height = height * 0.3
    
    # Create the cylindrical base
    base = primitive_call('cylinder', 
                         shape_kwargs={'radius': base_radius, 
                                      'p0': (0, 0, 0), 
                                      'p1': (0, base_height, 0)},
                         color=color)
    
    # Create the conical top
    top_base = primitive_call('cylinder', 
                             shape_kwargs={'radius': base_radius, 
                                          'p0': (0, base_height, 0), 
                                          'p1': (0, base_height + 0.01, 0)},
                             color=color)
    
    top_tip = primitive_call('cylinder', 
                            shape_kwargs={'radius': base_radius * 0.1, 
                                         'p0': (0, height - 0.01, 0), 
                                         'p1': (0, height, 0)},
                            color=color)
    
    # Create intermediate cylinders for the conical shape
    def loop_fn(i) -> Shape:
        progress = i / 10
        y_pos = base_height + progress * top_height
        radius = base_radius * (1 - progress)
        return primitive_call('cylinder', 
                             shape_kwargs={'radius': radius, 
                                          'p0': (0, y_pos, 0), 
                                          'p1': (0, y_pos + top_height/10, 0)},
                             color=color)
    
    cone_parts = loop(10, loop_fn)
    
    return concat_shapes(base, top_base, cone_parts, top_tip)

@register()
def decorative_element(size: float, color: tuple[float, float, float] = (0.75, 0.75, 0.65)) -> Shape:
    """Creates a decorative element for the towers"""
    sphere = primitive_call('sphere', shape_kwargs={'radius': size/2}, color=color)
    
    # Add some small spikes
    def loop_fn(i) -> Shape:
        angle = 2 * math.pi * i / 8
        spike = primitive_call('cylinder', 
                              shape_kwargs={'radius': size/10, 
                                           'p0': (0, 0, 0), 
                                           'p1': (size * math.cos(angle), size * math.sin(angle), 0)},
                              color=color)
        return spike
    
    spikes = loop(8, loop_fn)
    return concat_shapes(sphere, spikes)

@register()
def decorated_tower(height: float, base_radius: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a tower with decorative elements"""
    tower_shape = library_call('tower', height=height, base_radius=base_radius, color=color)
    
    # Add decorative elements along the tower
    def loop_fn(i) -> Shape:
        y_pos = height * 0.2 + i * (height * 0.5) / 5
        element = library_call('decorative_element', size=base_radius * 0.5, color=color)
        
        # Place elements around the tower
        def element_loop_fn(j) -> Shape:
            angle = 2 * math.pi * j / 4
            x = base_radius * 1.1 * math.cos(angle)
            z = base_radius * 1.1 * math.sin(angle)
            return transform_shape(element, translation_matrix((x, y_pos, z)))
        
        return loop(4, element_loop_fn)
    
    decorations = loop(5, loop_fn)
    
    # Add a decorative element at the top
    top_element = library_call('decorative_element', size=base_radius * 0.3, color=color)
    top_element = transform_shape(top_element, translation_matrix((0, height, 0)))
    
    return concat_shapes(tower_shape, decorations, top_element)

@register()
def main_body(width: float, length: float, height: float, color: tuple[float, float, float] = (0.85, 0.85, 0.75)) -> Shape:
    """Creates the main body of the basilica"""
    # Base structure
    base = primitive_call('cube', shape_kwargs={'scale': (width, height * 0.6, length)}, color=color)
    
    # Roof structure (slightly narrower)
    roof_width = width * 0.9
    roof_length = length * 0.9
    roof_height = height * 0.4
    
    roof = primitive_call('cube', shape_kwargs={'scale': (roof_width, roof_height, roof_length)}, color=color)
    roof = transform_shape(roof, translation_matrix((0, height * 0.6, 0)))
    
    # Add some texture to the roof
    def roof_detail_fn(i) -> Shape:
        x_pos = (i - 5) * (roof_width / 10)
        detail = primitive_call('cube', 
                               shape_kwargs={'scale': (roof_width/20, roof_height * 0.2, roof_length)}, 
                               color=color)
        return transform_shape(detail, translation_matrix((x_pos, height * 0.6 + roof_height * 0.5, 0)))
    
    roof_details = loop(11, roof_detail_fn)
    
    return concat_shapes(base, roof, roof_details)

@register()
def window(width: float, height: float, color: tuple[float, float, float] = (0.6, 0.8, 0.9)) -> Shape:
    """Creates a church window"""
    # Window frame
    frame = primitive_call('cube', shape_kwargs={'scale': (width, height, width * 0.1)}, color=(0.7, 0.7, 0.6))
    
    # Glass
    glass = primitive_call('cube', shape_kwargs={'scale': (width * 0.8, height * 0.8, width * 0.05)}, color=color)
    glass = transform_shape(glass, translation_matrix((0, 0, width * 0.05)))
    
    # Cross pattern on window
    h_cross = primitive_call('cube', shape_kwargs={'scale': (width * 0.8, height * 0.05, width * 0.15)}, color=(0.7, 0.7, 0.6))
    h_cross = transform_shape(h_cross, translation_matrix((0, 0, 0)))
    
    v_cross = primitive_call('cube', shape_kwargs={'scale': (width * 0.05, height * 0.8, width * 0.15)}, color=(0.7, 0.7, 0.6))
    v_cross = transform_shape(v_cross, translation_matrix((0, 0, 0)))
    
    return concat_shapes(frame, glass, h_cross, v_cross)

@register()
def entrance(width: float, height: float, depth: float, color: tuple[float, float, float] = (0.75, 0.75, 0.65)) -> Shape:
    """Creates an entrance for the basilica"""
    # Main arch
    base = primitive_call('cube', shape_kwargs={'scale': (width, height * 0.7, depth)}, color=color)
    
    # Arched top
    def arch_fn(i) -> Shape:
        progress = i / 10
        y_offset = height * 0.7 + progress * height * 0.3
        x_scale = width * (1 - progress)
        
        segment = primitive_call('cube', 
                                shape_kwargs={'scale': (x_scale, height * 0.03, depth)}, 
                                color=color)
        return transform_shape(segment, translation_matrix((0, y_offset, 0)))
    
    arch = loop(10, arch_fn)
    
    # Door
    door = primitive_call('cube', shape_kwargs={'scale': (width * 0.6, height * 0.6, depth * 0.1)}, color=(0.4, 0.3, 0.2))
    door = transform_shape(door, translation_matrix((0, height * 0.3, depth * 0.5)))
    
    # Decorative elements around the entrance
    def decor_fn(i) -> Shape:
        angle = math.pi * i / 10
        x = width * 0.5 * math.cos(angle)
        y = height * 0.7 + width * 0.5 * math.sin(angle)
        
        element = primitive_call('sphere', shape_kwargs={'radius': width * 0.05}, color=color)
        return transform_shape(element, translation_matrix((x, y, depth * 0.8)))
    
    decorations = loop(11, decor_fn)
    
    return concat_shapes(base, arch, door, decorations)

@register()
def sagrada_familia() -> Shape:
    """Creates the Basílica de la Sagrada Família"""
    # Main body
    main = library_call('main_body', width=5.0, length=8.0, height=3.0)
    
    # Towers
    def tower_fn(i) -> Shape:
        # Create 4 main towers at the corners
        if i < 4:
            x = 2.0 * (1 if i % 2 == 0 else -1)
            z = 3.5 * (1 if i < 2 else -1)
            height = 8.0 + np.random.uniform(-0.5, 0.5)
            tower = library_call('decorated_tower', height=height, base_radius=0.6)
            return transform_shape(tower, translation_matrix((x, 0, z)))
        # Create 4 smaller towers in the middle
        else:
            idx = i - 4
            x = 1.0 * (1 if idx % 2 == 0 else -1)
            z = 2.0 * (1 if idx < 2 else -1)
            height = 6.0 + np.random.uniform(-0.3, 0.3)
            tower = library_call('decorated_tower', height=height, base_radius=0.4)
            return transform_shape(tower, translation_matrix((x, 0, z)))
    
    towers = loop(8, tower_fn)
    
    # Central tower (tallest)
    central_tower = library_call('decorated_tower', height=10.0, base_radius=0.7)
    
    # Windows
    def window_fn(i) -> Shape:
        # Place windows along the sides of the main body
        side = i // 5
        pos = i % 5
        
        if side == 0:  # Front
            x = (pos - 2) * 0.8
            z = 4.0
            rotation = 0
        elif side == 1:  # Back
            x = (pos - 2) * 0.8
            z = -4.0
            rotation = math.pi
        elif side == 2:  # Left
            x = -2.5
            z = (pos - 2) * 0.8
            rotation = math.pi / 2
        else:  # Right
            x = 2.5
            z = (pos - 2) * 0.8
            rotation = -math.pi / 2
        
        window = library_call('window', width=0.6, height=1.2)
        window = transform_shape(window, rotation_matrix(rotation, (0, 1, 0), (0, 0, 0)))
        return transform_shape(window, translation_matrix((x, 1.5, z)))
    
    windows = loop(20, window_fn)
    
    # Entrances
    front_entrance = library_call('entrance', width=2.0, height=2.0, depth=0.8)
    front_entrance = transform_shape(front_entrance, translation_matrix((0, 0, 4.0)))
    
    back_entrance = library_call('entrance', width=2.0, height=2.0, depth=0.8)
    back_entrance = transform_shape(back_entrance, rotation_matrix(math.pi, (0, 1, 0), (0, 0, 0)))
    back_entrance = transform_shape(back_entrance, translation_matrix((0, 0, -4.0)))
    
    # Combine all elements
    return concat_shapes(
        main, 
        towers, 
        central_tower,
        windows,
        front_entrance,
        back_entrance
    )
```