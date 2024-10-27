"""
Here's a program to create a spider puppet using the provided `helper.py`:

"""
from helper import *

"""
a spider puppet
"""

@register()
def spider_body(radius: float) -> Shape:
    return primitive_call('sphere', color=(0.2, 0.2, 0.2), shape_kwargs={'radius': radius})

@register()
def spider_leg(length: float, radius: float) -> Shape:
    return primitive_call('cylinder', color=(0.1, 0.1, 0.1), shape_kwargs={'radius': radius, 'p0': (0, 0, 0), 'p1': (length, 0, 0)})

@register()
def spider_eye(radius: float) -> Shape:
    return primitive_call('sphere', color=(1.0, 0.0, 0.0), shape_kwargs={'radius': radius})

@register()
def spider_legs(body_radius: float, leg_length: float, leg_radius: float) -> Shape:
    def leg_pair(angle: float) -> Shape:
        leg = library_call('spider_leg', length=leg_length, radius=leg_radius)
        leg_left = transform_shape(leg, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        leg_right = transform_shape(leg, rotation_matrix(-angle, (0, 1, 0), (0, 0, 0)))
        return concat_shapes(
            transform_shape(leg_left, translation_matrix((body_radius, 0, 0))),
            transform_shape(leg_right, translation_matrix((-body_radius, 0, 0)))
        )

    angles = [math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3]
    return concat_shapes(*[leg_pair(angle) for angle in angles])

@register()
def spider_eyes(body_radius: float, eye_radius: float) -> Shape:
    eye = library_call('spider_eye', radius=eye_radius)
    eye_distance = body_radius * 0.4
    eye_height = body_radius * 0.7

    left_eye = transform_shape(eye, translation_matrix((eye_distance, eye_height, -body_radius * 0.8)))
    right_eye = transform_shape(eye, translation_matrix((-eye_distance, eye_height, -body_radius * 0.8)))

    return concat_shapes(left_eye, right_eye)

@register()
def spider_puppet(body_radius: float = 0.3, leg_length: float = 0.8, leg_radius: float = 0.02, eye_radius: float = 0.03) -> Shape:
    body = library_call('spider_body', radius=body_radius)
    legs = library_call('spider_legs', body_radius=body_radius, leg_length=leg_length, leg_radius=leg_radius)
    eyes = library_call('spider_eyes', body_radius=body_radius, eye_radius=eye_radius)

    puppet = concat_shapes(body, legs, eyes)

    # Add strings for the puppet
    string_length = 1.5
    string_radius = 0.005
    string = primitive_call('cylinder', color=(0.8, 0.8, 0.8), shape_kwargs={'radius': string_radius, 'p0': (0, 0, 0), 'p1': (0, string_length, 0)})

    left_string = transform_shape(string, translation_matrix((body_radius * 0.5, body_radius, 0)))
    right_string = transform_shape(string, translation_matrix((-body_radius * 0.5, body_radius, 0)))

    return concat_shapes(puppet, left_string, right_string)
"""

This program creates a spider puppet with the following components:

1. A spherical body
2. Eight legs (four on each side)
3. Two red eyes
4. Two strings attached to the body for puppet control

The `spider_puppet` function is the main function that combines all the parts. You can adjust the size of the spider by modifying the parameters of this function.

The spider's body is created using a black sphere. The legs are cylinders, positioned and rotated to give a realistic spider-like appearance. The eyes are small red spheres placed on the front of the body. Finally, two white strings (thin cylinders) are attached to the top of the body to represent the puppet strings.

You can create the spider puppet by calling:

```python
library_call('spider_puppet')
```

This will create a spider puppet with default dimensions. You can also customize the size by providing different parameters, for example:

```python
library_call('spider_puppet', body_radius=0.4, leg_length=1.0, leg_radius=0.025, eye_radius=0.04)
```

This will create a larger spider puppet with longer legs and bigger eyes.
"""