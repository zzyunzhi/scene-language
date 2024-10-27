from typing import Union

def unwrap_results(lines: list[str], code_only: bool = False) -> Union[list[str], None]:
    """
    Unwrap the results from the markdown file.
    Assume the code block is wrapped within ```python ... ```
    """
    lines = [line.rstrip() for line in lines]
    if '```python' not in lines:
        # print(lines)
        # import ipdb; ipdb.set_trace()
        return None
    if '```' not in lines:
        return None
        # import ipdb; ipdb.set_trace()
    code_start = lines.index('```python') + 1
    code_end = lines.index('```')
    # print(code_start, code_end)
    if code_only:
        return (lines[code_start:code_end]) 
    else:
        return (([] if code_start == 1 else ['"""'] +
                lines[:code_start - 1] +
                ['"""']) +
                lines[code_start:code_end] +
                ([] if code_end == (len(lines) - 1) else ['"""'] +
                lines[code_end + 1:] +
                ['"""']))


if __name__ == "__main__":
    test = ['Here is a possible implementation of a colorful Jenga tower in Minecraft using the provided DSL:\n\n```python\nfrom helper import *\n\n"""\na colorful Jenga tower\n"""\n\n@register(\'a colorful Jenga tower\')\ndef jenga_tower(height: int = 18, width: int = 3) -> Shape:\n    return loop(height, lambda i, n: transform_shape(\n        library_call(\'jenga_layer\', width=width, color=BlockType(jenga_color(i))), \n        translation_matrix([0, i, 0])\n    ))\n\n@register(\'a single layer of a Jenga tower\')\ndef jenga_layer(width: int = 3, color: BlockType = BlockType.PLANKS) -> Shape:\n    return primitive_call(\'set_cuboid\', block_type=color, scale=(width, 1, width), fill=True)\n\ndef jenga_color(i: int) -> str:\n    colors = [\'red_wool\', \'orange_wool\', \'yellow_wool\', \'lime_wool\', \'light_blue_wool\', \'magenta_wool\']\n    return colors[i % len(colors)]\n```\n\nExplanation:\n- The main `jenga_tower` function takes a `height` and `width` parameter to specify the dimensions of the tower. It uses a `loop` to stack `height` number of `jenga_layer`s on top of each other, each translated upwards by their index `i`.\n- The `jenga_layer` function creates a single solid layer of the tower with the specified `width` and `color`. It uses `primitive_call` to create a filled cuboid.\n- The `jenga_color` helper function returns a different wool color based on the index `i`, cycling through a predefined list of colors. This is used to make each layer of the tower a different color.\n\nThe result is a tower with the specified `height` and `width`, composed of solid layers each with a different vibrant color. The layers are stacked directly on top of each other to form the tower structure.\n\nLet me know if you have any other questions!']
    test = test[0].split('\n')
    print(unwrap_results(test))