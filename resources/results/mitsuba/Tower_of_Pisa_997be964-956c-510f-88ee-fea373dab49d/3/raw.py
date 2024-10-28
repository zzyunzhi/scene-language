"""
I'll help you create a model of the Tower of Pisa with its characteristic tilt and cylindrical architecture. Here's the implementation:

"""
from helper import *

"""
Tower of Pisa
"""

@register()
def cylinder_level(radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    """Creates a single cylindrical level of the tower"""
    return primitive_call('cylinder',
                        shape_kwargs={'radius': radius,
                                    'p0': (0, 0, 0),
                                    'p1': (0, height, 0)},
                        color=color)

@register()
def columns(radius: float, height: float, num_columns: int) -> Shape:
    """Creates decorative columns around the main cylinder"""
    def loop_fn(i: int) -> Shape:
        angle = (2 * math.pi * i) / num_columns
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        column = primitive_call('cylinder',
                              shape_kwargs={'radius': 0.05,
                                          'p0': (x, 0, z),
                                          'p1': (x, height, z)},
                              color=(0.9, 0.9, 0.85))
        return column

    return loop(num_columns, loop_fn)

@register()
def tower_level(level_radius: float, height: float, with_columns: bool = True) -> Shape:
    """Creates a complete level of the tower with main cylinder and optional columns"""
    main_cylinder = library_call('cylinder_level',
                               radius=level_radius,
                               height=height,
                               color=(0.95, 0.95, 0.9))

    if not with_columns:
        return main_cylinder

    columns_shape = library_call('columns',
                               radius=level_radius + 0.05,
                               height=height,
                               num_columns=12)

    return concat_shapes(main_cylinder, columns_shape)

@register()
def tower_of_pisa() -> Shape:
    """Creates the complete Tower of Pisa with its characteristic tilt"""
    # Base parameters
    base_radius = 1.0
    level_height = 1.0
    tilt_angle = -0.1  # tilt angle in radians

    # Create different levels with decreasing radius
    levels = []
    for i in range(8):
        level_radius = base_radius - (i * 0.05)
        level = library_call('tower_level',
                           level_radius=level_radius,
                           height=level_height,
                           with_columns=(i > 0))  # no columns on base

        # Stack levels and apply cumulative transformations
        level_translation = translation_matrix((0, i * level_height, 0))
        level_center = (0, i * level_height + level_height/2, 0)
        level_rotation = rotation_matrix(tilt_angle, (1, 0, 0), level_center)

        transformed_level = transform_shape(level, level_translation)
        transformed_level = transform_shape(transformed_level, level_rotation)
        levels.append(transformed_level)

    # Add a bell tower at the top
    bell_tower = library_call('tower_level',
                            level_radius=base_radius - 0.4,
                            height=level_height * 0.7,
                            with_columns=True)

    # Transform bell tower
    bell_translation = translation_matrix((0, 8 * level_height, 0))
    bell_center = (0, 8 * level_height + (level_height * 0.7)/2, 0)
    bell_rotation = rotation_matrix(tilt_angle, (1, 0, 0), bell_center)

    transformed_bell = transform_shape(bell_tower, bell_translation)
    transformed_bell = transform_shape(transformed_bell, bell_rotation)
    levels.append(transformed_bell)

    return concat_shapes(*levels)
"""

This implementation creates a simplified version of the Tower of Pisa with these features:

1. The tower consists of multiple cylindrical levels with decreasing radius as it goes up
2. Each level (except the base) has decorative columns around its perimeter
3. The characteristic tilt is achieved by rotating each level slightly around the x-axis
4. The tower has 8 main levels plus a bell tower at the top
5. The color scheme uses off-white tones typical of marble construction
6. The architecture includes:
   - A solid base level without columns
   - Multiple levels with decorative columns
   - A smaller bell tower at the top
   - Gradually decreasing radius as the tower goes up
   - The famous lean implemented through cumulative rotations

The tower is built from bottom to top, with each level being positioned and then tilted appropriately. The tilt is consistent throughout the structure, creating the iconic leaning effect.
"""