"""
Here's a program to reconstruct the input scene using the provided `helper.py`:

"""
from helper import *

"""
Reconstruct the input scene
"""

@register()
def vase(radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    return primitive_call('cylinder', shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=color)

@register()
def sphere(radius: float, color: tuple[float, float, float]) -> Shape:
    return primitive_call('sphere', shape_kwargs={'radius': radius}, color=color)

@register()
def table() -> Shape:
    return primitive_call('cube', shape_kwargs={'scale': (2, 0.05, 1)}, color=(0.6, 0.4, 0.2))

@register()
def round_vase() -> Shape:
    base = library_call('vase', radius=0.15, height=0.3, color=(1.0, 0.8, 0.6))
    top = library_call('sphere', radius=0.15, color=(1.0, 0.8, 0.6))
    return concat_shapes(
        base,
        transform_shape(top, translation_matrix((0, 0.3, 0)))
    )

@register()
def yellow_vase() -> Shape:
    return library_call('vase', radius=0.2, height=0.4, color=(1.0, 0.8, 0.0))

@register()
def pink_tall_vase() -> Shape:
    return library_call('vase', radius=0.1, height=0.6, color=(1.0, 0.6, 0.8))

@register()
def purple_vase() -> Shape:
    return library_call('vase', radius=0.15, height=0.4, color=(0.8, 0.4, 0.8))

@register()
def small_sphere() -> Shape:
    return library_call('sphere', radius=0.05, color=(0.9, 0.9, 0.7))

@register()
def scene() -> Shape:
    table_top = library_call('table')
    round_vase_shape = transform_shape(library_call('round_vase'), translation_matrix((-0.7, 0.05, 0)))
    yellow_vase_shape = transform_shape(library_call('yellow_vase'), translation_matrix((-0.3, 0.05, 0)))
    pink_tall_vase_shape = transform_shape(library_call('pink_tall_vase'), translation_matrix((0.1, 0.05, 0)))
    purple_vase_shape = transform_shape(library_call('purple_vase'), translation_matrix((0.5, 0.05, 0)))
    small_sphere_shape = transform_shape(library_call('small_sphere'), translation_matrix((0.3, 0.05, 0.2)))

    return concat_shapes(
        table_top,
        round_vase_shape,
        yellow_vase_shape,
        pink_tall_vase_shape,
        purple_vase_shape,
        small_sphere_shape
    )
"""

This program creates a scene that approximates the image provided. Here's a breakdown of the functions:

1. `vase`: A generic function to create cylindrical vases.
2. `sphere`: A function to create spheres.
3. `table`: Creates the table surface.
4. `round_vase`: Creates the rounded vase on the left.
5. `yellow_vase`: Creates the yellow vase.
6. `pink_tall_vase`: Creates the tall pink vase in the middle.
7. `purple_vase`: Creates the purple vase on the right.
8. `small_sphere`: Creates the small sphere in the foreground.
9. `scene`: Assembles all the elements into the final scene.

The `scene` function positions all the vases and the small sphere on the table surface. The positioning is approximate based on the image. You may need to adjust the translation values to get the exact positioning as in the image.

Note that this reconstruction is a simplified 3D representation and doesn't include details like the flowers or the textured background. It focuses on the main shapes and their approximate colors and positions.
"""