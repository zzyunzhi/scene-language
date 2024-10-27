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
    bowl = primitive_call('sphere', shape_kwargs={'radius': radius}, color=color)
    return transform_shape(bowl, translation_matrix((0, height, 0)))

@register()
def dessert_glass(stem_height: float, stem_radius: float, bowl_radius: float, color: tuple[float, float, float]) -> Shape:
    stem = library_call('glass_stem', height=stem_height, radius=stem_radius, color=color)
    bowl = library_call('glass_bowl', radius=bowl_radius, height=stem_height, color=color)
    return concat_shapes(stem, bowl)

@register()
def ice_cream_scoop(radius: float, color: tuple[float, float, float]) -> Shape:
    return primitive_call('sphere', shape_kwargs={'radius': radius}, color=color)

@register()
def cherry(radius: float) -> Shape:
    cherry_body = primitive_call('sphere', shape_kwargs={'radius': radius}, color=(0.8, 0.1, 0.1))
    stem = primitive_call('cylinder', shape_kwargs={'radius': radius * 0.1, 'p0': (0, 0, 0), 'p1': (0, radius * 2, 0)}, color=(0.5, 0.3, 0.1))
    return concat_shapes(cherry_body, stem)

@register()
def sundae_glass() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.15, stem_radius=0.02, bowl_radius=0.1, color=(0.9, 0.9, 1.0))
    ice_cream = library_call('ice_cream_scoop', radius=0.08, color=(0.95, 0.95, 0.8))
    cherry = library_call('cherry', radius=0.02)

    glass_height = compute_shape_max(glass)[1]
    ice_cream = transform_shape(ice_cream, translation_matrix((0, glass_height + 0.03, 0)))
    cherry = transform_shape(cherry, translation_matrix((0, glass_height + 0.11, 0)))

    return concat_shapes(glass, ice_cream, cherry)

@register()
def layered_dessert_glass() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.2, stem_radius=0.02, bowl_radius=0.08, color=(0.9, 0.9, 1.0))

    def create_layer(height: float, color: tuple[float, float, float]) -> Shape:
        layer = primitive_call('cylinder', shape_kwargs={'radius': 0.075, 'p0': (0, 0, 0), 'p1': (0, 0.03, 0)}, color=color)
        return transform_shape(layer, translation_matrix((0, height, 0)))

    layer1 = create_layer(0.2, (0.2, 0.8, 0.2))  # Green layer
    layer2 = create_layer(0.23, (0.9, 0.7, 0.7))  # Pink layer

    return concat_shapes(glass, layer1, layer2)

@register()
def striped_dessert_glass() -> Shape:
    glass = library_call('dessert_glass', stem_height=0.18, stem_radius=0.02, bowl_radius=0.07, color=(0.9, 0.9, 1.0))

    def create_stripe(height: float, color: tuple[float, float, float]) -> Shape:
        stripe = primitive_call('cylinder', shape_kwargs={'radius': 0.065, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)}, color=color)
        return transform_shape(stripe, translation_matrix((0, height, 0)))

    stripes = [create_stripe(0.18 + i * 0.025, (1.0, 0.7, 0.0) if i % 2 == 0 else (0.9, 0.9, 0.9)) for i in range(5)]

    return concat_shapes(glass, *stripes)

@register()
def ice_cream_bowl() -> Shape:
    bowl = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(0.8, 0.4, 0.0))
    bowl = transform_shape(bowl, scale_matrix(0.7, (0, 0, 0)))
    bowl = transform_shape(bowl, translation_matrix((0, 0.05, 0)))

    ice_cream = library_call('ice_cream_scoop', radius=0.07, color=(0.95, 0.95, 0.8))
    ice_cream = transform_shape(ice_cream, translation_matrix((0, 0.1, 0)))

    cherry = library_call('cherry', radius=0.015)
    cherry = transform_shape(cherry, translation_matrix((0, 0.17, 0)))

    return concat_shapes(bowl, ice_cream, cherry)

@register()
def dessert_scene() -> Shape:
    sundae = library_call('sundae_glass')
    layered = library_call('layered_dessert_glass')
    striped = library_call('striped_dessert_glass')
    bowl = library_call('ice_cream_bowl')

    sundae = transform_shape(sundae, translation_matrix((-0.3, 0, 0)))
    layered = transform_shape(layered, translation_matrix((-0.1, 0, 0)))
    striped = transform_shape(striped, translation_matrix((0.1, 0, 0)))
    bowl = transform_shape(bowl, translation_matrix((0.3, 0, 0)))

    return concat_shapes(sundae, layered, striped, bowl)