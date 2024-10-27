import numpy as np

import colorsys


def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    return colorsys.rgb_to_hls(r, g, b)


def describe_color(color: tuple[float, float, float]) -> str:
    if not all(0 <= c <= 1 for c in color):
        print(f'[ERROR] invalid {color=}')
        return None

    color_dict = {
        (0.8, .4, .3): 'muted reddish',
        (.9, .7, .5): 'light peachy orange',
        (.6, .3, .1): 'dark brown',
        (0.4, 0.2, 0.1): 'dark brown',
        (.9, .9, .9): 'light gray',
        (.1, .1, .1): 'dark gray',
    }
    if tuple(color) in color_dict:
        return color_dict[(tuple(color))]

    r, g, b = color
    h, l, s = rgb_to_hsl(r, g, b)

    # Convert hue to degrees
    h_deg = h * 360

    # Determine hue name
    hue_names = [
        (0, "red"), (30, "orange"), (60, "yellow"), (90, "yellow-green"),
        (120, "green"), (180, "cyan"), (210, "sky blue"), (240, "blue"),
        (270, "purple"), (300, "magenta"), (330, "pink"), (360, "red")
    ]
    hue = next((name for deg, name in hue_names if h_deg <= deg), "red")

    # Determine saturation
    if s < 0.1:
        saturation = "gray"
    elif s < 0.3:
        saturation = "muted"
    elif s > 0.8:
        saturation = "vivid"
    else:
        saturation = ""

    # Determine lightness
    if l < 0.2:
        lightness = "very dark"
    elif l < 0.4:
        lightness = "dark"
    elif l > 0.8:
        lightness = "very light"
    elif l > 0.6:
        lightness = "light"
    else:
        lightness = ""

    # Special cases for white, black, and grays
    if l > 0.95:
        return "white"
    elif l < 0.05:
        return "black"
    elif s < 0.1:
        if l < 0.2:
            return "very dark gray"
        elif l < 0.4:
            return "dark gray"
        elif l > 0.8:
            return "very light gray"
        elif l > 0.6:
            return "light gray"
        else:
            return "gray"

    # Combine descriptions
    descriptions = [d for d in [lightness, saturation, hue] if d]
    return " ".join(descriptions)


if __name__ == "__main__":
    colors = [(1, 1, 1), (0.6, 0.3, 0.1), (0.8, 0.8, 0.2), (0.2, 0.6, 0.8)]
    for color in colors:
        print(color, describe_color(color))
