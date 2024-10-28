"""
I'll help you create a program to generate the Tower of Pisa with its characteristic tilt and cylindrical architecture. Here's the implementation:

"""
from helper import *

"""
Tower of Pisa
"""

@register()
def tower_level(radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    """Creates a single level of the tower with a cylindrical body and decorative columns"""
    main_body = primitive_call('cylinder',
                             shape_kwargs={'radius': radius,
                                         'p0': (0, 0, 0),
                                         'p1': (0, height, 0)},
                             color=color)

    def column_fn(i: int) -> Shape:
        angle = i * 2 * math.pi / 8  # 8 columns around the level
        x = (radius - 0.1) * math.cos(angle)
        z = (radius - 0.1) * math.sin(angle)
        return primitive_call('cylinder',
                            shape_kwargs={'radius': 0.08,
                                        'p0': (x, 0, z),
                                        'p1': (x, height, z)},
                            color=(0.9, 0.9, 0.85))

    columns = loop(8, column_fn)
    return concat_shapes(main_body, columns)

@register()
def tower_top() -> Shape:
    """Creates the bell tower top with a smaller radius"""
    return library_call('tower_level', radius=2.0, height=3.0, color=(0.95, 0.95, 0.9))

@register()
def tower_base() -> Shape:
    """Creates the wider base of the tower"""
    return library_call('tower_level', radius=4.0, height=4.0, color=(0.9, 0.9, 0.85))

@register()
def tower_middle_section() -> Shape:
    """Creates the middle sections of the tower"""
    def level_fn(i: int) -> Shape:
        level = library_call('tower_level', radius=3.0, height=3.0, color=(0.93, 0.93, 0.88))
        return transform_shape(level, translation_matrix((0, i * 3.0, 0)))

    return loop(6, level_fn)

@register()
def tower_of_pisa() -> Shape:
    """Assembles the complete Tower of Pisa with its characteristic tilt"""
    # Create base
    base = library_call('tower_base')

    # Create middle section
    middle = library_call('tower_middle_section')
    middle = transform_shape(middle, translation_matrix((0, 4.0, 0)))

    # Create top
    top = library_call('tower_top')
    top = transform_shape(top, translation_matrix((0, 22.0, 0)))

    # Combine all sections
    tower = concat_shapes(base, middle, top)

    # Apply the famous tilt (approximately 3.99 degrees)
    tower_center = compute_shape_center(tower)
    tilt_angle = -0.0696  # radians (≈ 3.99 degrees)
    tilted_tower = transform_shape(tower, rotation_matrix(tilt_angle, (0, 0, 1), tower_center))

    # Add ground for context
    ground = primitive_call('cylinder',
                          shape_kwargs={'radius': 8.0,
                                      'p0': (0, -0.5, 0),
                                      'p1': (0, 0, 0)},
                          color=(0.7, 0.8, 0.6))

    return concat_shapes(ground, tilted_tower)
"""

This implementation creates the Tower of Pisa with these key features:

1. `tower_level`: Creates a single level of the tower with a cylindrical main body and 8 decorative columns around it.

2. `tower_base`: Creates a wider base section.

3. `tower_middle_section`: Creates 6 identical middle sections.

4. `tower_top`: Creates the bell tower top section with a smaller radius.

5. `tower_of_pisa`: Assembles all components and:
   - Stacks the sections vertically
   - Applies the famous tilt (approximately 3.99 degrees)
   - Adds a circular ground base
   - Uses realistic proportions and colors

The tower features:
- Cylindrical architecture with decorative columns
- Gradually decreasing radius from base to top
- The characteristic tilt
- Multiple levels with consistent styling
- A ground base for context
- Realistic coloring using slightly different shades of off-white/cream

The implementation uses all the required helper functions and follows the coordinate system (+x right, +y up, +z backward). The tower is created modularly, making it easy to modify individual components or adjust the overall appearance.
"""