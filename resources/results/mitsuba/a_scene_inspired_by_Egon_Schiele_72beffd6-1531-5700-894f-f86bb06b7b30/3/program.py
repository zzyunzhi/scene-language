"""
Here's a program inspired by Egon Schiele's art style, focusing on elongated figures and expressive poses:

"""
from helper import *

"""
a scene inspired by Egon Schiele
"""

@register()
def elongated_limb(length: float, width: float) -> Shape:
    return primitive_call('cylinder', color=(0.9, 0.8, 0.7), shape_kwargs={
        'radius': width / 2,
        'p0': (0, 0, 0),
        'p1': (0, length, 0)
    })

@register()
def torso(height: float, width: float, depth: float) -> Shape:
    return primitive_call('cube', color=(0.9, 0.8, 0.7), shape_kwargs={
        'scale': (width, height, depth)
    })

@register()
def head() -> Shape:
    return primitive_call('sphere', color=(0.9, 0.8, 0.7), shape_kwargs={
        'radius': 0.15
    })

@register()
def schiele_figure() -> Shape:
    # Create body parts
    body = library_call('torso', height=0.8, width=0.3, depth=0.2)
    head_shape = library_call('head')
    arm = library_call('elongated_limb', length=0.6, width=0.08)
    leg = library_call('elongated_limb', length=0.9, width=0.1)

    # Position head
    head_shape = transform_shape(head_shape, translation_matrix((0, 0.9, 0)))

    # Position arms
    left_arm = transform_shape(arm, rotation_matrix(math.pi/6, (0, 0, 1), (0, 0, 0)))
    left_arm = transform_shape(left_arm, translation_matrix((0.2, 0.7, 0)))
    right_arm = transform_shape(arm, rotation_matrix(-math.pi/4, (0, 0, 1), (0, 0, 0)))
    right_arm = transform_shape(right_arm, translation_matrix((-0.2, 0.7, 0)))

    # Position legs
    left_leg = transform_shape(leg, translation_matrix((0.1, -0.4, 0)))
    right_leg = transform_shape(leg, rotation_matrix(math.pi/12, (0, 0, 1), (0, 0, 0)))
    right_leg = transform_shape(right_leg, translation_matrix((-0.1, -0.4, 0)))

    return concat_shapes(body, head_shape, left_arm, right_arm, left_leg, right_leg)

@register()
def schiele_composition() -> Shape:
    figure1 = library_call('schiele_figure')
    figure2 = library_call('schiele_figure')
    figure3 = library_call('schiele_figure')

    # Position figures
    figure1 = transform_shape(figure1, translation_matrix((-0.5, -0.5, 0)))
    figure2 = transform_shape(figure2, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    figure2 = transform_shape(figure2, translation_matrix((0.5, -0.3, 0.3)))
    figure3 = transform_shape(figure3, rotation_matrix(-math.pi/4, (0, 1, 0), (0, 0, 0)))
    figure3 = transform_shape(figure3, translation_matrix((0, 0.5, -0.2)))

    return concat_shapes(figure1, figure2, figure3)

@register()
def schiele_scene() -> Shape:
    composition = library_call('schiele_composition')

    # Create a simple background
    background = primitive_call('cube', color=(0.9, 0.85, 0.8), shape_kwargs={'scale': (3, 3, 0.1)})
    background = transform_shape(background, translation_matrix((0, 0, 0.5)))

    return concat_shapes(background, composition)
"""

This program creates a scene inspired by Egon Schiele's art style, known for its elongated figures and expressive poses. Here's a breakdown of the functions:

1. `elongated_limb`: Creates a cylindrical limb with adjustable length and width.
2. `torso`: Creates a rectangular torso.
3. `head`: Creates a spherical head.
4. `schiele_figure`: Assembles a figure using the body parts, positioning them to create an expressive pose characteristic of Schiele's style.
5. `schiele_composition`: Creates a composition of three figures, each positioned and rotated differently to create an interesting arrangement.
6. `schiele_scene`: Adds a simple background to the composition.

The scene captures key elements of Schiele's style:
- Elongated limbs and angular poses
- Multiple figures in a composition
- Simple, flat background

To generate the scene, you would call `library_call('schiele_scene')`.
"""