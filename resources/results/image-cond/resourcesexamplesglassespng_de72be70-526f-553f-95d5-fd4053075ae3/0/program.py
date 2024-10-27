from helper import *

"""
Reconstruct the input scene of desserts in glasses
"""

@register()
def glass_stem(height: float, base_radius: float, stem_radius: float) -> Shape:
    base = primitive_call('cylinder', shape_kwargs={'radius': base_radius, 'p0': (0, 0, 0), 'p1': (0, 0.05, 0)}, color=(0.9, 0.9, 0.9))
    stem = primitive_call('cylinder', shape_kwargs={'radius': stem_radius, 'p0': (0, 0.05, 0), 'p1': (0, height, 0)}, color=(0.9, 0.9, 0.9))
    return concat_shapes(base, stem)

@register()
def glass_bowl(radius: float, height: float) -> Shape:
    return primitive_call('cylinder', shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=(0.9, 0.9, 0.95))

@register()
def dessert_glass(stem_height: float, bowl_radius: float, bowl_height: float) -> Shape:
    stem = library_call('glass_stem', height=stem_height, base_radius=bowl_radius*0.7, stem_radius=bowl_radius*0.1)
    bowl = library_call('glass_bowl', radius=bowl_radius, height=bowl_height)
    return concat_shapes(stem, transform_shape(bowl, translation_matrix((0, stem_height, 0))))

@register()
def dessert_filling(radius: float, height: float, color: tuple[float, float, float]) -> Shape:
    return primitive_call('cylinder', shape_kwargs={'radius': radius*0.9, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=color)

@register()
def cherry() -> Shape:
    cherry_body = primitive_call('sphere', shape_kwargs={'radius': 0.02}, color=(0.8, 0.1, 0.1))
    cherry_stem = primitive_call('cylinder', shape_kwargs={'radius': 0.002, 'p0': (0, 0, 0), 'p1': (0, 0.04, 0)}, color=(0.4, 0.2, 0.1))
    return concat_shapes(cherry_body, transform_shape(cherry_stem, translation_matrix((0, 0.02, 0))))

@register()
def dessert_1() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.15, bowl_radius=0.06, bowl_height=0.12)
    filling = library_call('dessert_filling', radius=0.06, height=0.1, color=(0.6, 0.3, 0.2))
    whipped_cream = library_call('dessert_filling', radius=0.06, height=0.03, color=(0.95, 0.95, 0.95))
    cherry = library_call('cherry')

    return concat_shapes(
        glass,
        transform_shape(filling, translation_matrix((0, 0.15, 0))),
        transform_shape(whipped_cream, translation_matrix((0, 0.25, 0))),
        transform_shape(cherry, translation_matrix((0, 0.28, 0)))
    )

@register()
def dessert_2() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.2, bowl_radius=0.05, bowl_height=0.15)
    filling_1 = library_call('dessert_filling', radius=0.05, height=0.05, color=(0.9, 0.6, 0.7))
    filling_2 = library_call('dessert_filling', radius=0.05, height=0.05, color=(0.3, 0.8, 0.4))
    whipped_cream = library_call('dessert_filling', radius=0.05, height=0.06, color=(0.95, 0.95, 0.95))
    cherry = library_call('cherry')

    return concat_shapes(
        glass,
        transform_shape(filling_1, translation_matrix((0, 0.2, 0))),
        transform_shape(filling_2, translation_matrix((0, 0.25, 0))),
        transform_shape(whipped_cream, translation_matrix((0, 0.3, 0))),
        transform_shape(cherry, translation_matrix((0, 0.36, 0)))
    )

@register()
def dessert_3() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.18, bowl_radius=0.055, bowl_height=0.14)

    def create_layer(height: float, color: tuple[float, float, float]) -> Shape:
        return library_call('dessert_filling', radius=0.055, height=height, color=color)

    layers = loop(5, lambda i: create_layer(0.02, (0.9, 0.7 + i*0.05, 0.3)))
    whipped_cream = library_call('dessert_filling', radius=0.055, height=0.04, color=(0.95, 0.95, 0.95))
    cherry = library_call('cherry')

    dessert = concat_shapes(glass, *[transform_shape(layer, translation_matrix((0, 0.18 + i*0.02, 0))) for i, layer in enumerate(layers)])
    dessert = concat_shapes(dessert, transform_shape(whipped_cream, translation_matrix((0, 0.28, 0))))
    dessert = concat_shapes(dessert, transform_shape(cherry, translation_matrix((0, 0.32, 0))))

    return dessert

@register()
def dessert_4() -> Shape:
    bowl = library_call('glass_bowl', radius=0.08, height=0.06)
    ice_cream = primitive_call('sphere', shape_kwargs={'radius': 0.07}, color=(0.95, 0.95, 0.9))
    cherry = library_call('cherry')

    return concat_shapes(
        bowl,
        transform_shape(ice_cream, translation_matrix((0, 0.07, 0))),
        transform_shape(cherry, translation_matrix((0, 0.14, 0)))
    )

@register()
def dessert_scene() -> Shape:
    dessert1 = transform_shape(library_call('dessert_1'), translation_matrix((-0.18, 0, 0)))
    dessert2 = transform_shape(library_call('dessert_2'), translation_matrix((-0.06, 0, 0)))
    dessert3 = transform_shape(library_call('dessert_3'), translation_matrix((0.06, 0, 0)))
    dessert4 = transform_shape(library_call('dessert_4'), translation_matrix((0.18, 0, 0)))

    return concat_shapes(dessert1, dessert2, dessert3, dessert4)