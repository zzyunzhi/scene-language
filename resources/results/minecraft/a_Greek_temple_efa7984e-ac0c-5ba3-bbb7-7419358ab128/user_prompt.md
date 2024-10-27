Here are some examples of how to use `helper.py`:
```python
from helper import *

"""
A red cube on the top left of a blue pyramid of height 4.
"""


@register()
def cube_set() -> Shape:
    return concat_shapes(
        library_call(
            "red_cube"
        ),  # expects a cube with left-bottom-front corner block at (-2, 7, 2) and dims 2x2x2
        library_call("blue_pyramid"),  # expects a blue pyramid of height 4
    )  # hint: these library calls must be implemented to be compatible with the usage


@register()
def red_cube() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:redstone_block",
            scale=(2, 2, 2),
            fill=True,
        ),
        translation_matrix([-2, 7, 2]),
    )


@register()
def blue_pyramid(n: int = 4) -> Shape:
    def create_pyramid_layer(i):
        # Logic here is that for the ith layer, it has dims (2*i + 1) x1x(2*i + 1.
        # We need to then shift that in the x dimension to center it, and then also in the y dimension to lift to the right layer of the pyramid.
        side_length = i * 2 + 1
        last_layer_length = n * 2 + 1
        x_z_offset = (last_layer_length - side_length) // 2
        y_offset = n - i - 1
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:lapis_block",
                scale=(side_length, 1, side_length),
                fill=True,
            ),
            translation_matrix([x_z_offset, y_offset, x_z_offset]),
        )

    return loop(4, create_pyramid_layer)


"""
A forest of trees of varying heights.
"""


@register()
def forest(leaf_size: int = 3) -> Shape:
    # Double for loop for placing the trees
    tree_padding = (
        leaf_size * 2 + 3
    )  # This is how far the center point of each tree should be from each other
    return loop(
        4,
        lambda i: loop(
            4,
            lambda j: transform_shape(
                library_call(
                    "simple_tree", height=random.randint(3, 7)
                ),  # Make it random to give the appearance of having varying heights
                translation_matrix(
                    [i * leaf_size + tree_padding, 0, j * leaf_size + tree_padding]
                ),
            ),
        ),
    )


@register()
def simple_tree(height: int = 4) -> Shape:
    return concat_shapes(
        library_call("trunk", trunk_height=height),
        transform_shape(
            library_call(
                "leaves", leaf_size=3
            ),  # If you pass in extra arguments to library_call, they need to be NAMED arguments. Passing in 3 here without "leaf_size" will error.
            translation_matrix(
                [-1, height, -1]
            ),  # Center the leaves on top of the trunk
        ),
    )


@register()
def leaves(leaf_size: int = 3) -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:oak_leaves",
        block_kwargs={"distance": "7", "persistent": "true", "waterlogged": "false"},
        scale=(leaf_size, leaf_size, leaf_size),
        fill=True,
    )


@register()
def trunk(trunk_height: int = 4) -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:oak_log",
        block_kwargs={"axis": "y"},
        scale=(1, trunk_height, 1),
        fill=True,
    )



```
IMPORTANT: THE FUNCTIONS ABOVE ARE JUST EXAMPLES, YOU CANNOT USE THEM IN YOUR PROGRAM! 

Now, write a similar program for the given task:    
```python
from helper import *

"""
a Greek temple
"""
```
