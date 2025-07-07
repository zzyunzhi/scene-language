Your task is to review the following Python code and provide detailed feedback on (ordered by importance):
- Code correctness, particularly the usage of the provided DSL. The current proposal can be properly executed and rendered! Look for other issues.
- Whether the generated 3D scene matches the described task and common sense. Carefully examine the provided image(s) from different viewpoints rendered from the current proposal. For EACH function output, check if the object is in the right position and orientation. A typical failure mode is translation missing by half of the object size!! Note that the camera is automatically positioned to see the whole scene. Include error analysis in your review.
- Only if everything else is correct, improve on scene details and aesthetics. 

Task description:
Basílica de la Sagrada Família

Here is the current code proposal from the writer:
```python
from helper import *

"""
Basílica de la Sagrada Família
"""

@register()
def tower(height: float, base_radius: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a tower with a conical top and Gaudí-inspired details"""
    base_height = height * 0.7
    top_height = height * 0.3

    # Create the cylindrical base
    base = primitive_call('cylinder',
                         shape_kwargs={'radius': base_radius,
                                      'p0': (0, 0, 0),
                                      'p1': (0, base_height, 0)},
                         color=color)

    # Create the conical top with more intricate detailing
    def loop_fn(i) -> Shape:
        progress = i / 10
        y_pos = base_height + progress * top_height
        radius = base_radius * (1 - progress)

        # Add texture to the spire
        segment = primitive_call('cylinder',
                               shape_kwargs={'radius': radius,
                                            'p0': (0, y_pos, 0),
                                            'p1': (0, y_pos + top_height/10, 0)},
                               color=color)

        # Add small decorative elements around each segment
        if i % 2 == 0 and i > 0:
            def detail_fn(j) -> Shape:
                angle = 2 * math.pi * j / 6
                x = radius * 1.1 * math.cos(angle)
                z = radius * 1.1 * math.sin(angle)
                detail = primitive_call('sphere',
                                      shape_kwargs={'radius': radius * 0.2},
                                      color=(color[0]*0.9, color[1]*0.9, color[2]*0.9))
                return transform_shape(detail, translation_matrix((x, y_pos, z)))

            details = loop(6, detail_fn)
            return concat_shapes(segment, details)

        return segment

    cone_parts = loop(10, loop_fn)

    # Add a decorative top
    top_element = primitive_call('sphere', shape_kwargs={'radius': base_radius * 0.2}, color=color)
    top_element = transform_shape(top_element, translation_matrix((0, height, 0)))

    return concat_shapes(base, cone_parts, top_element)

@register()
def decorative_element(size: float, color: tuple[float, float, float] = (0.75, 0.75, 0.65)) -> Shape:
    """Creates a decorative element for the towers with spikes in all directions"""
    sphere = primitive_call('sphere', shape_kwargs={'radius': size/2}, color=color)

    # Add spikes in all directions (not just XY plane)
    def loop_fn(i) -> Shape:
        if i < 8:  # XY plane
            angle = 2 * math.pi * i / 8
            spike = primitive_call('cylinder',
                                 shape_kwargs={'radius': size/10,
                                              'p0': (0, 0, 0),
                                              'p1': (size * math.cos(angle), size * math.sin(angle), 0)},
                                 color=color)
        elif i < 12:  # XZ plane
            angle = 2 * math.pi * (i-8) / 4
            spike = primitive_call('cylinder',
                                 shape_kwargs={'radius': size/10,
                                              'p0': (0, 0, 0),
                                              'p1': (size * math.cos(angle), 0, size * math.sin(angle))},
                                 color=color)
        else:  # YZ plane
            angle = 2 * math.pi * (i-12) / 4
            spike = primitive_call('cylinder',
                                 shape_kwargs={'radius': size/10,
                                              'p0': (0, 0, 0),
                                              'p1': (0, size * math.cos(angle), size * math.sin(angle))},
                                 color=color)
        return spike

    spikes = loop(16, loop_fn)
    return concat_shapes(sphere, spikes)

@register()
def decorated_tower(height: float, base_radius: float, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates a tower with decorative elements integrated into the structure"""
    tower_shape = library_call('tower', height=height, base_radius=base_radius, color=color)

    # Add decorative elements along the tower
    def loop_fn(i) -> Shape:
        y_pos = height * 0.2 + i * (height * 0.5) / 5
        element_size = base_radius * 0.5
        element = library_call('decorative_element', size=element_size, color=color)

        # Place elements around the tower, integrated with the structure
        def element_loop_fn(j) -> Shape:
            angle = 2 * math.pi * j / 4
            x = base_radius * 0.9 * math.cos(angle)  # Moved closer to tower surface
            z = base_radius * 0.9 * math.sin(angle)
            return transform_shape(element, translation_matrix((x, y_pos, z)))

        return loop(4, element_loop_fn)

    decorations = loop(5, loop_fn)

    # Add a decorative element at the top
    top_element = library_call('decorative_element', size=base_radius * 0.3, color=color)
    top_element = transform_shape(top_element, translation_matrix((0, height, 0)))

    return concat_shapes(tower_shape, decorations, top_element)

@register()
def main_body(width: float, length: float, height: float, color: tuple[float, float, float] = (0.85, 0.85, 0.75)) -> Shape:
    """Creates the main body of the basilica with a cruciform layout"""
    # Base structure - cruciform layout
    main_nave = primitive_call('cube', shape_kwargs={'scale': (width, height * 0.6, length)}, color=color)

    # Transept (crossing section)
    transept = primitive_call('cube', shape_kwargs={'scale': (length * 0.8, height * 0.6, width * 0.8)}, color=color)

    # Roof structure (slightly narrower)
    roof_width = width * 0.9
    roof_length = length * 0.9
    roof_height = height * 0.4

    roof = primitive_call('cube', shape_kwargs={'scale': (roof_width, roof_height, roof_length)}, color=color)
    roof = transform_shape(roof, translation_matrix((0, height * 0.6, 0)))

    # Transept roof
    transept_roof = primitive_call('cube',
                                  shape_kwargs={'scale': (roof_length * 0.8, roof_height, roof_width * 0.8)},
                                  color=color)
    transept_roof = transform_shape(transept_roof, translation_matrix((0, height * 0.6, 0)))

    # Add some texture to the roof
    def roof_detail_fn(i) -> Shape:
        x_pos = (i - 5) * (roof_width / 10)
        detail = primitive_call('cube',
                               shape_kwargs={'scale': (roof_width/20, roof_height * 0.2, roof_length)},
                               color=color)
        return transform_shape(detail, translation_matrix((x_pos, height * 0.6 + roof_height * 0.5, 0)))

    roof_details = loop(11, roof_detail_fn)

    return concat_shapes(main_nave, transept, roof, transept_roof, roof_details)

@register()
def window(width: float, height: float, color: tuple[float, float, float] = (0.6, 0.8, 0.9)) -> Shape:
    """Creates a church window with stained glass effect"""
    # Window frame
    frame = primitive_call('cube', shape_kwargs={'scale': (width, height, width * 0.1)}, color=(0.7, 0.7, 0.6))

    # Glass
    glass = primitive_call('cube', shape_kwargs={'scale': (width * 0.8, height * 0.8, width * 0.05)}, color=color)
    glass = transform_shape(glass, translation_matrix((0, 0, width * 0.05)))

    # Cross pattern on window - properly aligned with glass
    h_cross = primitive_call('cube', shape_kwargs={'scale': (width * 0.8, height * 0.05, width * 0.15)}, color=(0.7, 0.7, 0.6))
    h_cross = transform_shape(h_cross, translation_matrix((0, 0, width * 0.05)))

    v_cross = primitive_call('cube', shape_kwargs={'scale': (width * 0.05, height * 0.8, width * 0.15)}, color=(0.7, 0.7, 0.6))
    v_cross = transform_shape(v_cross, translation_matrix((0, 0, width * 0.05)))

    # Add some decorative elements to make it more Gaudí-like
    def detail_fn(i) -> Shape:
        angle = 2 * math.pi * i / 8
        radius = width * 0.4
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        detail = primitive_call('sphere', shape_kwargs={'radius': width * 0.05}, color=(0.7, 0.7, 0.6))
        return transform_shape(detail, translation_matrix((x, y, width * 0.05)))

    details = loop(8, detail_fn)

    return concat_shapes(frame, glass, h_cross, v_cross, details)

@register()
def entrance(width: float, height: float, depth: float, color: tuple[float, float, float] = (0.75, 0.75, 0.65)) -> Shape:
    """Creates an elaborate entrance for the basilica"""
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
def ground_plane(size: float = 20.0, color: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> Shape:
    """Creates a ground plane"""
    return primitive_call('cube', shape_kwargs={'scale': (size, 0.1, size)}, color=color)

@register()
def facade(width: float, height: float, depth: float, facade_type: str, color: tuple[float, float, float] = (0.8, 0.8, 0.7)) -> Shape:
    """Creates one of the three grand façades (Nativity, Passion, or Glory)"""
    # Base structure
    base = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Different decorative elements based on façade type
    if facade_type == "nativity":
        # More organic, nature-inspired elements
        def detail_fn(i) -> Shape:
            x = (i % 5 - 2) * (width * 0.2)
            y = (i // 5) * (height * 0.25) + height * 0.3
            size = width * 0.08
            detail = library_call('decorative_element', size=size, color=color)
            return transform_shape(detail, translation_matrix((x, y, depth * 0.5)))

        details = loop(15, detail_fn)

    elif facade_type == "passion":
        # More angular, severe elements
        def detail_fn(i) -> Shape:
            x = (i % 3 - 1) * (width * 0.3)
            y = (i // 3) * (height * 0.3) + height * 0.2
            cube = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, width * 0.15, depth * 0.2)}, color=color)
            return transform_shape(cube, translation_matrix((x, y, depth * 0.5)))

        details = loop(9, detail_fn)

    else:  # "glory"
        # Grand, triumphant elements
        def detail_fn(i) -> Shape:
            angle = 2 * math.pi * i / 12
            radius = width * 0.4
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) + height * 0.5
            sphere = primitive_call('sphere', shape_kwargs={'radius': width * 0.06}, color=color)
            return transform_shape(sphere, translation_matrix((x, y, depth * 0.5)))

        details = loop(12, detail_fn)

    # Add an entrance to each façade
    entrance_shape = library_call('entrance', width=width * 0.6, height=height * 0.6, depth=depth * 0.3)
    entrance_shape = transform_shape(entrance_shape, translation_matrix((0, 0, depth * 0.5)))

    return concat_shapes(base, details, entrance_shape)

@register()
def sagrada_familia() -> Shape:
    """Creates the Basílica de la Sagrada Família"""
    # Main body
    main = library_call('main_body', width=5.0, length=8.0, height=3.0)

    # Ground plane
    ground = library_call('ground_plane')
    ground = transform_shape(ground, translation_matrix((0, -0.05, 0)))  # Position slightly below the structure

    # Towers - 12 for Apostles, 4 for Evangelists, 1 for Mary, 1 for Jesus
    def tower_fn(i) -> Shape:
        if i < 4:  # Four main towers at corners (Evangelists)
            x = 2.0 * (1 if i % 2 == 0 else -1)
            z = 3.5 * (1 if i < 2 else -1)
            height = 8.0 + np.random.uniform(-0.5, 0.5)
            tower = library_call('decorated_tower', height=height, base_radius=0.6)
            # Position towers on top of the main body
            return transform_shape(tower, translation_matrix((x, 1.5, z)))
        elif i < 16:  # 12 Apostle towers around the perimeter
            idx = i - 4
            angle = 2 * math.pi * idx / 12
            radius = 3.0
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            height = 6.0 + np.random.uniform(-0.3, 0.3)
            tower = library_call('decorated_tower', height=height, base_radius=0.4)
            return transform_shape(tower, translation_matrix((x, 1.5, z)))
        elif i == 16:  # Mary's tower
            tower = library_call('decorated_tower', height=7.0, base_radius=0.5)
            return transform_shape(tower, translation_matrix((0, 1.5, 2.0)))
        else:  # Jesus's tower (central, tallest)
            tower = library_call('decorated_tower', height=10.0, base_radius=0.7)
            return transform_shape(tower, translation_matrix((0, 1.5, 0)))  # Position on top of main body

    towers = loop(18, tower_fn)

    # Windows - positioned relative to main body height
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
        # Position windows at appropriate height on main body
        return transform_shape(window, translation_matrix((x, 1.5, z)))

    windows = loop(20, window_fn)

    # Three grand façades
    nativity_facade = library_call('facade', width=4.0, height=4.0, depth=1.0, facade_type="nativity")
    nativity_facade = transform_shape(nativity_facade, translation_matrix((0, 1.5, 4.0)))

    passion_facade = library_call('facade', width=4.0, height=4.0, depth=1.0, facade_type="passion")
    passion_facade = transform_shape(passion_facade, rotation_matrix(math.pi, (0, 1, 0), (0, 0, 0)))
    passion_facade = transform_shape(passion_facade, translation_matrix((0, 1.5, -4.0)))

    glory_facade = library_call('facade', width=4.0, height=4.0, depth=1.0, facade_type="glory")
    glory_facade = transform_shape(glory_facade, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    glory_facade = transform_shape(glory_facade, translation_matrix((2.5, 1.5, 0)))

    # Combine all elements
    return concat_shapes(
        ground,
        main,
        towers,
        windows,
        nativity_facade,
        passion_facade,
        glory_facade
    )
```

Provide your critiques and suggestions for improvement below in a formatted list.


![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250303-234310_3a199e3d-04e3-424e-99c3-394e4540615c/Basilica_de_la_Sagrada_Familia_20fa601b-6d24-557a-a9cf-ff686568f4fe/expert_00_refl_02_writer/renderings/exposed_sagrada_familia_rover_background_frame_00/rendering_traj_000.png)

