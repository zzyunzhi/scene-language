"""
Here's a program to create an octopus sculpture using the provided `helper.py`:

"""

from helper import *

"""
octopus sculpture
"""


@register()
def octopus_head(radius: float) -> Shape:
    return primitive_call(
        "sphere", color=(0.5, 0.2, 0.5), shape_kwargs={"radius": radius}
    )


@register()
def octopus_eye(radius: float) -> Shape:
    eye_white = primitive_call(
        "sphere", color=(1.0, 1.0, 1.0), shape_kwargs={"radius": radius}
    )
    eye_pupil = primitive_call(
        "sphere", color=(0.0, 0.0, 0.0), shape_kwargs={"radius": radius * 0.5}
    )
    return concat_shapes(
        eye_white, transform_shape(eye_pupil, translation_matrix((0, 0, radius * 0.5)))
    )


@register()
def octopus_tentacle(length: float, radius: float, segments: int) -> Shape:
    def segment(i: int) -> Shape:
        segment_length = length / segments
        segment_radius = radius * (1 - i / segments)
        cylinder = primitive_call(
            "cylinder",
            color=(0.5, 0.2, 0.5),
            shape_kwargs={
                "radius": segment_radius,
                "p0": (0, 0, 0),
                "p1": (0, 0, segment_length),
            },
        )
        angle = math.sin(i * math.pi / 4) * 0.5
        rotation = rotation_matrix(angle, (1, 0, 0), (0, 0, 0))
        translation = translation_matrix((0, 0, i * segment_length))
        return transform_shape(cylinder, translation @ rotation)

    return loop(segments, segment)


@register()
def octopus() -> Shape:
    head = library_call("octopus_head", radius=1.0)

    left_eye = library_call("octopus_eye", radius=0.2)
    left_eye = transform_shape(left_eye, translation_matrix((-0.5, 0.5, -0.8)))

    right_eye = library_call("octopus_eye", radius=0.2)
    right_eye = transform_shape(right_eye, translation_matrix((0.5, 0.5, -0.8)))

    def create_tentacle(angle: float) -> Shape:
        tentacle = library_call("octopus_tentacle", length=4.0, radius=0.2, segments=8)
        rotation = rotation_matrix(angle, (0, 1, 0), (0, 0, 0))
        translation = translation_matrix((0, -1, 0))
        return transform_shape(tentacle, translation @ rotation)

    tentacles = [create_tentacle(i * math.pi / 4) for i in range(8)]

    return concat_shapes(head, left_eye, right_eye, *tentacles)


"""

This program creates an octopus sculpture with the following components:

1. A spherical head
2. Two eyes, each composed of a white sphere and a black pupil
3. Eight tentacles, each made up of cylindrical segments that taper and curve

The `octopus()` function assembles all these components into a complete octopus sculpture. The head is centered at the origin, the eyes are positioned on the front of the head, and the tentacles are arranged radially around the bottom of the head.

You can create the octopus sculpture by calling `library_call('octopus')`.
"""


##### END HERE ######


if __name__ == "__main__":
    from impl import set_seed

    set_seed(1)

    import json

    from dsl_utils import library, set_lock_enabled
    from shape_utils import create_hole

    from mi_helper import execute
    from pathlib import Path
    from mitsuba import Point3f

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                # Convert NumPy arrays to lists
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                # Convert NumPy floats to Python floats
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                # Convert NumPy integers to Python ints
                return int(obj)
            elif isinstance(obj, Point3f):
                # Convert Point3f to a list [x, y, z]
                return [obj.x, obj.y, obj.z]
            # Handle other types as needed
            return super().default(obj)

    nodes = []
    target_render_name = "octopus"
    for name in list(library.keys()):
        if name in [node.name for node in nodes]:
            continue
        node: "Hole" = create_hole(
            name=name,
            docstring=library[name]["docstring"],
            check=library[name]["check"],
        )
        node.implement(lambda: library[name]["__target__"])
        nodes.append(node)

        if name == target_render_name:
            # _ = node()  # dummy run to register all functions
            print(f"Rendering {name}")
            shapes = []
            objs = node()
            for obj in objs:
                shape = {
                    "type": obj["type"],  # Example: 'cylinder'
                    "properties": {},
                    "to_world": obj[
                        "to_world"
                    ].tolist(),  # Convert numpy array to regular Python list
                }

                # Handle different object types and properties
                if obj["type"] == "cylinder":
                    shape["properties"]["radiusTop"] = obj["radius"]
                    shape["properties"]["radiusBottom"] = obj["radius"]

                    # Include p0 and p1 in the JSON
                    shape["properties"]["p0"] = obj["p0"]
                    shape["properties"]["p1"] = obj["p1"]

                    # Calculate and set the height of the cylinder
                    shape["properties"]["height"] = np.linalg.norm(
                        np.array(obj["p1"]) - np.array(obj["p0"])
                    )

                    # Extract color from 'bsdf'
                    if (
                        obj["bsdf"]["type"] == "diffuse"
                        and "reflectance" in obj["bsdf"]
                    ):
                        reflectance = obj["bsdf"]["reflectance"]
                        if reflectance["type"] == "rgb":
                            shape["properties"]["color"] = reflectance[
                                "value"
                            ].tolist()  # Convert numpy array to list
                elif obj["type"] == "sphere":
                    shape["properties"]["radius"] = 1
                    if (
                        obj["bsdf"]["type"] == "diffuse"
                        and "reflectance" in obj["bsdf"]
                    ):
                        reflectance = obj["bsdf"]["reflectance"]
                        if reflectance["type"] == "rgb":
                            shape["properties"]["color"] = reflectance[
                                "value"
                            ].tolist()  # Convert numpy array to list
                elif obj["type"] == "cuboid":
                    raise RuntimeError("Unimplemented")

                # Add to shapes list
                shapes.append(shape)

            # Define the path to the JSON file
            json_file_path = "/Users/matthewzhou/Projects/engine/scripts/minecraft/threejs-viewer/src/static/example.json"

            # Write the JSON data to the file
            with open(json_file_path, "w") as json_file:
                json.dump(
                    {"shapes": shapes}, json_file, indent=4, cls=CustomJSONEncoder
                )

            print(f"JSON data successfully written to {json_file_path}")
