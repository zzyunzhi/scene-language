from helper import *

"""
Reconstruct the input scene
"""

@register()
def coca_cola_can() -> Shape:
    # Create the main body of the can
    body = primitive_call('cylinder', shape_kwargs={'radius': 0.033, 'p0': (0, 0, 0), 'p1': (0, 0.122, 0)}, color=(0.8, 0, 0))

    # Create the top and bottom lids
    lid_radius = 0.033
    lid_height = 0.005
    top_lid = primitive_call('cylinder', shape_kwargs={'radius': lid_radius, 'p0': (0, 0.122, 0), 'p1': (0, 0.122 + lid_height, 0)}, color=(0.8, 0.8, 0.8))
    bottom_lid = primitive_call('cylinder', shape_kwargs={'radius': lid_radius, 'p0': (0, 0, 0), 'p1': (0, -lid_height, 0)}, color=(0.8, 0.8, 0.8))

    return concat_shapes(body, top_lid, bottom_lid)

@register()
def coca_cola_pack(num_cans: int, rows: int) -> Shape:
    def create_can(i: int) -> Shape:
        row = i // (num_cans // rows)
        col = i % (num_cans // rows)
        can = library_call('coca_cola_can')
        x_offset = col * 0.07
        z_offset = row * 0.07
        return transform_shape(can, translation_matrix((x_offset, 0, z_offset)))

    return loop(num_cans, create_can)

@register()
def scene() -> Shape:
    pack = library_call('coca_cola_pack', num_cans=6, rows=2)

    # Adjust the position of the pack
    pack_center = compute_shape_center(pack)
    adjusted_pack = transform_shape(pack, translation_matrix((-pack_center[0], -pack_center[1], -pack_center[2])))

    return adjusted_pack