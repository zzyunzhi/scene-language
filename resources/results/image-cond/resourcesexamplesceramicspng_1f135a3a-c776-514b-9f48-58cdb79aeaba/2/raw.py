"""
Here's a program to reconstruct the input scene using the provided `helper.py`:

"""
from helper import *

"""
Reconstruct the input scene
"""

@register()
def spherical_vase(radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    sphere = primitive_call('sphere', shape_kwargs={'radius': radius}, color=color)
    cylinder = primitive_call('cylinder', shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=color)
    return concat_shapes(sphere, cylinder)

@register()
def bottle_vase(base_radius: float, neck_radius: float, height: float, neck_height: float, color: tuple[float, float, float]) -> Shape:
    base = primitive_call('cylinder', shape_kwargs={'radius': base_radius, 'p0': (0, 0, 0), 'p1': (0, height - neck_height, 0)}, color=color)
    neck = primitive_call('cylinder', shape_kwargs={'radius': neck_radius, 'p0': (0, height - neck_height, 0), 'p1': (0, height, 0)}, color=color)
    return concat_shapes(base, neck)

@register()
def curvy_vase(base_radius: float, max_radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    def create_section(y: float, radius: float) -> Shape:
        return primitive_call('cylinder', shape_kwargs={'radius': radius, 'p0': (0, y, 0), 'p1': (0, y + height/20, 0)}, color=color)

    sections = loop(20, lambda i: create_section(i * height/20, base_radius + (max_radius - base_radius) * math.sin(i * math.pi / 20)))
    return concat_shapes(*sections)

@register()
def table() -> Shape:
    table_top = primitive_call('cube', shape_kwargs={'scale': (2, 0.05, 1)}, color=(0.8, 0.6, 0.4))
    return table_top

@register()
def scene() -> Shape:
    table_shape = library_call('table')

    spherical_vase_shape = library_call('spherical_vase', radius=0.15, height=0.2, color=(0.9, 0.6, 0.5))
    bottle_vase_yellow = library_call('bottle_vase', base_radius=0.1, neck_radius=0.05, height=0.4, neck_height=0.1, color=(1, 0.8, 0))
    bottle_vase_pink = library_call('bottle_vase', base_radius=0.08, neck_radius=0.04, height=0.5, neck_height=0.2, color=(0.9, 0.5, 0.7))
    curvy_vase = library_call('curvy_vase', base_radius=0.06, max_radius=0.1, height=0.4, color=(0.8, 0.4, 0.6))

    small_sphere = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.9, 0.9, 0.8))

    table_top = compute_shape_max(table_shape)[1]

    return concat_shapes(
        table_shape,
        transform_shape(spherical_vase_shape, translation_matrix((0.6, table_top, 0.3))),
        transform_shape(bottle_vase_yellow, translation_matrix((-0.2, table_top, 0.2))),
        transform_shape(bottle_vase_pink, translation_matrix((0.2, table_top, 0.2))),
        transform_shape(curvy_vase, translation_matrix((0.5, table_top, 0.2))),
        transform_shape(small_sphere, translation_matrix((0, table_top, 0.4)))
    )
"""

This program creates a scene that approximates the input image. Here's a breakdown of the components:

1. `spherical_vase`: Creates a spherical vase with a cylindrical body.
2. `bottle_vase`: Creates a bottle-shaped vase with a wider base and narrower neck.
3. `curvy_vase`: Creates a vase with a curved profile using multiple cylindrical sections.
4. `table`: Creates a simple table top.
5. `scene`: Assembles all the components into the final scene.

The scene consists of:
- A table
- A spherical vase (pink/orange one on the left)
- Two bottle-shaped vases (yellow one in the middle and pink one on the right)
- A curvy vase (purple one on the far right)
- A small sphere (representing the small round object in the foreground)

The program doesn't include the plant elements inside the vases, as those would require more complex geometry that's beyond the scope of the primitive shapes available. The colors and positions are approximated based on the image.
"""