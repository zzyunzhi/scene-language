"""
Here's a Python program to reconstruct the input scene using the provided `helper.py`:

"""
from helper import *

"""
Reconstruct the input scene of four dessert glasses
"""

@register()
def glass_stem(height: float, radius: float, color: tuple[float, float, float]) -> Shape:
    stem = primitive_call('cylinder', shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=color)
    base = primitive_call('cylinder', shape_kwargs={'radius': radius * 2, 'p0': (0, 0, 0), 'p1': (0, height * 0.1, 0)}, color=color)
    return concat_shapes(stem, base)

@register()
def glass_bowl(radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    return primitive_call('cylinder', shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=color)

@register()
def dessert_glass(stem_height: float, stem_radius: float, bowl_radius: float, bowl_height: float, stem_color: tuple[float, float, float], bowl_color: tuple[float, float, float]) -> Shape:
    stem = library_call('glass_stem', height=stem_height, radius=stem_radius, color=stem_color)
    bowl = library_call('glass_bowl', radius=bowl_radius, height=bowl_height, color=bowl_color)
    bowl = transform_shape(bowl, translation_matrix((0, stem_height, 0)))
    return concat_shapes(stem, bowl)

@register()
def chocolate_dessert() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.6, stem_radius=0.05, bowl_radius=0.2, bowl_height=0.4, stem_color=(0.8, 0.8, 0.8), bowl_color=(0.9, 0.9, 0.9))
    filling = primitive_call('cylinder', shape_kwargs={'radius': 0.18, 'p0': (0, 0.6, 0), 'p1': (0, 0.9, 0)}, color=(0.4, 0.2, 0.1))
    cream = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(1, 1, 0.9))
    cream = transform_shape(cream, translation_matrix((0, 0.95, 0)))
    cherry = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.8, 0.1, 0.1))
    cherry = transform_shape(cherry, translation_matrix((0, 1.05, 0)))
    return concat_shapes(glass, filling, cream, cherry)

@register()
def mint_dessert() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.7, stem_radius=0.03, bowl_radius=0.15, bowl_height=0.5, stem_color=(0.8, 0.8, 0.8), bowl_color=(0.9, 0.9, 0.9))
    mint_filling = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 0.7, 0), 'p1': (0, 1.0, 0)}, color=(0.2, 0.8, 0.4))
    cream_filling = primitive_call('cylinder', shape_kwargs={'radius': 0.13, 'p0': (0, 1.0, 0), 'p1': (0, 1.15, 0)}, color=(0.9, 0.7, 0.7))
    cherry = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.8, 0.1, 0.1))
    cherry = transform_shape(cherry, translation_matrix((0, 1.2, 0)))
    return concat_shapes(glass, mint_filling, cream_filling, cherry)

@register()
def vanilla_caramel_dessert() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.65, stem_radius=0.04, bowl_radius=0.17, bowl_height=0.6, stem_color=(0.8, 0.8, 0.8), bowl_color=(0.9, 0.9, 0.9))
    vanilla_filling = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0.65, 0), 'p1': (0, 1.05, 0)}, color=(1, 0.95, 0.8))
    caramel_swirl = primitive_call('cylinder', shape_kwargs={'radius': 0.01, 'p0': (0, 0.65, 0), 'p1': (0, 1.05, 0)}, color=(0.8, 0.4, 0.1))
    caramel_swirl = transform_shape(caramel_swirl, rotation_matrix(math.pi/4, (0, 1, 0), (0, 0.85, 0)))
    cherry = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.8, 0.1, 0.1))
    cherry = transform_shape(cherry, translation_matrix((0, 1.15, 0)))
    return concat_shapes(glass, vanilla_filling, caramel_swirl, cherry)

@register()
def ice_cream_bowl() -> Shape:
    bowl = primitive_call('cylinder', shape_kwargs={'radius': 0.2, 'p0': (0, 0, 0), 'p1': (0, 0.15, 0)}, color=(0.9, 0.7, 0.3))
    base = primitive_call('cylinder', shape_kwargs={'radius': 0.1, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)}, color=(0.9, 0.7, 0.3))
    ice_cream = primitive_call('sphere', shape_kwargs={'radius': 0.18}, color=(1, 1, 0.9))
    ice_cream = transform_shape(ice_cream, translation_matrix((0, 0.25, 0)))
    cherry = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.8, 0.1, 0.1))
    cherry = transform_shape(cherry, translation_matrix((0, 0.4, 0)))
    return concat_shapes(bowl, base, ice_cream, cherry)

@register()
def dessert_scene() -> Shape:
    chocolate = library_call('chocolate_dessert')
    mint = library_call('mint_dessert')
    vanilla_caramel = library_call('vanilla_caramel_dessert')
    ice_cream = library_call('ice_cream_bowl')

    chocolate = transform_shape(chocolate, translation_matrix((-0.6, 0, 0)))
    mint = transform_shape(mint, translation_matrix((-0.2, 0, 0)))
    vanilla_caramel = transform_shape(vanilla_caramel, translation_matrix((0.2, 0, 0)))
    ice_cream = transform_shape(ice_cream, translation_matrix((0.6, 0, 0)))

    return concat_shapes(chocolate, mint, vanilla_caramel, ice_cream)
"""

This program reconstructs the input scene of four dessert glasses. Here's a breakdown of the functions:

1. `glass_stem` and `glass_bowl`: These functions create the basic components of a glass.

2. `dessert_glass`: This function combines the stem and bowl to create a complete glass.

3. `chocolate_dessert`, `mint_dessert`, `vanilla_caramel_dessert`, and `ice_cream_bowl`: These functions create the four different desserts seen in the image.

4. `dessert_scene`: This function arranges the four desserts side by side to recreate the entire scene.

The program uses various primitives (cylinders and spheres) to approximate the shapes in the image. It also uses transformations to position the components correctly. The colors are approximated based on the image, using RGB values in the range [0, 1].

To render the scene, you would call `library_call('dessert_scene')`.
"""