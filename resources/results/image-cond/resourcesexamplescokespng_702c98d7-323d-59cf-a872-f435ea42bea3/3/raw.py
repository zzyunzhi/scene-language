from helper import *

"""
Reconstruct the input scene
"""

@register()
def soda_can(color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Shape:
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 0.15, 0)}, color=color)
    top = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0.15, 0), 'p1': (0, 0.16, 0)}, color=(0.8, 0.8, 0.8))
    return concat_shapes(body, top)

@register()
def soda_can_pack(num_cans: int, rows: int) -> Shape:
    def can_placement(i: int) -> Shape:
        row = i // (num_cans // rows)
        col = i % (num_cans // rows)
        x_offset = col * 0.11 - ((num_cans // rows - 1) * 0.11 / 2)
        z_offset = row * 0.11 - ((rows - 1) * 0.11 / 2)
        can = library_call('soda_can')
        return transform_shape(can, translation_matrix((x_offset, 0, z_offset)))

    return loop(num_cans, can_placement)

@register()
def coca_cola_logo(scale: float = 1.0) -> Shape:
    # Simplified representation of the Coca-Cola logo using primitives
    base = primitive_call('cube', shape_kwargs={'scale': (0.2 * scale, 0.05 * scale, 0.01 * scale)}, color=(0.8, 0, 0))
    circle1 = primitive_call('sphere', shape_kwargs={'radius': 0.025 * scale}, color=(0, 0, 0))
    circle2 = primitive_call('sphere', shape_kwargs={'radius': 0.025 * scale}, color=(0, 0, 0))

    logo = concat_shapes(
        base,
        transform_shape(circle1, translation_matrix((-0.05 * scale, 0.01 * scale, 0.005 * scale))),
        transform_shape(circle2, translation_matrix((0.05 * scale, 0.01 * scale, 0.005 * scale)))
    )

    return logo

@register()
def coca_cola_can_pack() -> Shape:
    can_pack = library_call('soda_can_pack', num_cans=6, rows=2)
    logo = library_call('coca_cola_logo', scale=0.5)

    # Apply logo to each can
    def apply_logo(i: int) -> Shape:
        can = transform_shape(library_call('soda_can'), translation_matrix((i * 0.11 - 0.165, 0, 0 if i < 3 else 0.11)))
        can_center = compute_shape_center(can)
        logo_transformed = transform_shape(logo, translation_matrix((can_center[0], can_center[1], can_center[2] + 0.05)))
        return concat_shapes(can, logo_transformed)

    cans_with_logos = loop(6, apply_logo)

    return cans_with_logos