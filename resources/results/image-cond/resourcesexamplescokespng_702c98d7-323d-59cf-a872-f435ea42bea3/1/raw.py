from helper import *

"""
Reconstruct the input scene
"""

@register()
def soda_can(color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> Shape:
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 0.15, 0)}, color=color)
    top = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0.15, 0), 'p1': (0, 0.155, 0)}, color=(0.8, 0.8, 0.8))
    bottom = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, -0.005, 0)}, color=(0.8, 0.8, 0.8))
    return concat_shapes(body, top, bottom)

@register()
def soda_can_pack(num_cans: int, rows: int) -> Shape:
    cans_per_row = num_cans // rows
    def create_row(row: int) -> Shape:
        def create_can(i: int) -> Shape:
            can = library_call('soda_can')
            x_offset = (i - (cans_per_row - 1) / 2) * 0.11
            z_offset = (row - (rows - 1) / 2) * 0.11
            return transform_shape(can, translation_matrix((x_offset, 0, z_offset)))
        return concat_shapes(*[create_can(i) for i in range(cans_per_row)])

    return concat_shapes(*[create_row(row) for row in range(rows)])

@register()
def coca_cola_pack() -> Shape:
    pack = library_call('soda_can_pack', num_cans=6, rows=2)
    return transform_shape(pack, translation_matrix((0, 0.075, 0)))
