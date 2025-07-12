"""
I'll help you create a Jenga tower with randomly displaced blocks. Here's the implementation:

"""
from helper import *

"""
A Jenga building with slightly displaced blocks
"""

@register()
def jenga_block(scale: P = (0.75, 0.15, 0.25)) -> Shape:
    """Creates a single Jenga block with light wooden color"""
    return primitive_call('cube',
                        color=(0.76, 0.6, 0.42),  # light wooden color
                        shape_kwargs={'scale': scale})

@register()
def jenga_layer(is_horizontal: bool) -> Shape:
    """Creates a layer of 3 Jenga blocks, either horizontally or vertically aligned"""
    def block_fn(i: int) -> Shape:
        block = library_call('jenga_block')

        # Calculate offset based on orientation
        if is_horizontal:
            offset = (0, 0, (i-1) * 0.25)  # Spread blocks along z-axis
            rotation = 0
        else:
            offset = ((i-1) * 0.25, 0, 0)  # Spread blocks along x-axis
            rotation = math.pi/2  # 90 degrees

        # First translate the block
        block = transform_shape(block, translation_matrix(offset))

        # Then rotate if needed (for vertical orientation)
        if not is_horizontal:
            block_center = compute_shape_center(block)
            block = transform_shape(block,
                                 rotation_matrix(rotation, (0, 1, 0), block_center))

        # Add small random displacement for realism
        random_offset = (
            np.random.uniform(-0.02, 0.02),
            0,
            np.random.uniform(-0.02, 0.02)
        )
        block = transform_shape(block, translation_matrix(random_offset))

        # Add slight random rotation for realism
        block_center = compute_shape_center(block)
        random_angle = np.random.uniform(-0.02, 0.02)
        return transform_shape(block,
                             rotation_matrix(random_angle, (0, 1, 0), block_center))

    return loop(3, block_fn)

@register()
def jenga_tower(num_layers: int = 18) -> Shape:
    """Creates a Jenga tower with the specified number of layers"""
    def layer_fn(i: int) -> Shape:
        is_horizontal = (i % 2 == 0)  # Alternate between horizontal and vertical layers
        layer = library_call('jenga_layer', is_horizontal=is_horizontal)

        # Stack layers vertically
        return transform_shape(layer,
                             translation_matrix((0, i * 0.15, 0)))  # 0.15 is block height

    return loop(num_layers, layer_fn)
"""

This implementation creates a realistic Jenga tower with:

1. Individual Jenga blocks with proper proportions (3:1:0.6 ratio)
2. Alternating layers (horizontal and vertical)
3. Three blocks per layer
4. Slight random displacements and rotations for realism
5. A light wooden color

The tower is built from bottom to top, with each layer consisting of three blocks. The blocks in each layer are either:
- Aligned along the z-axis (horizontal layers)
- Aligned along the x-axis (vertical layers, rotated 90 degrees)

For added realism, each block has:
- Small random displacement from its perfect position
- Slight random rotation around the vertical axis

You can create a Jenga tower by calling:
```python
library_call('jenga_tower')  # Creates default 18-layer tower
# or
library_call('jenga_tower', num_layers=12)  # Creates a shorter tower
```
"""