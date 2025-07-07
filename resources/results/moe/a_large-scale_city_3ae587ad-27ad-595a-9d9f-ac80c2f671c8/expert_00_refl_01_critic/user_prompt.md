Your task is to review the following Python code and provide detailed feedback on (ordered by importance):
- Code correctness, particularly the usage of the provided DSL. The current proposal can be properly executed and rendered! Look for other issues.
- Whether the generated 3D scene matches the described task and common sense. Carefully examine the provided image(s) from different viewpoints rendered from the current proposal. For EACH function output, check if the object is in the right position and orientation. A typical failure mode is translation missing by half of the object size!! Note that the camera is automatically positioned to see the whole scene. Include error analysis in your review.
- Only if everything else is correct, improve on scene details and aesthetics. 

Task description:
a large-scale city

Here is the current code proposal from the writer:
```python
from helper import *

"""
a large-scale city
"""

@register("Creates a building with specified dimensions and color")
def building(width: float, height: float, depth: float, color: tuple = (0.7, 0.7, 0.7)) -> Shape:
    return primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Add windows
    windows = []
    window_width = width * 0.15
    window_height = window_width * 1.5
    window_depth = 0.01
    window_color = (0.9, 0.9, 1.0)

    # Calculate number of windows per side
    windows_per_width = max(2, int(width / (window_width * 1.5)))
    windows_per_height = max(3, int(height / (window_height * 1.5)))

    # Function to create windows on one face
    def create_face_windows(face_direction: str) -> Shape:
        face_windows = []

        if face_direction in ['front', 'back']:
            z_pos = depth/2 if face_direction == 'front' else -depth/2
            x_spacing = width / (windows_per_width + 1)
            y_spacing = height / (windows_per_height + 1)

            for i in range(1, windows_per_width + 1):
                for j in range(1, windows_per_height + 1):
                    x_pos = (i * x_spacing) - width/2
                    y_pos = (j * y_spacing) - height/2 + height/2
                    window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
                    window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
                    face_windows.append(window)

        elif face_direction in ['left', 'right']:
            x_pos = width/2 if face_direction == 'right' else -width/2
            z_spacing = depth / (windows_per_width + 1)
            y_spacing = height / (windows_per_height + 1)

            for i in range(1, windows_per_width + 1):
                for j in range(1, windows_per_height + 1):
                    z_pos = (i * z_spacing) - depth/2
                    y_pos = (j * y_spacing) - height/2 + height/2
                    window = primitive_call('cube', shape_kwargs={'scale': (window_depth, window_height, window_width)}, color=window_color)
                    window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
                    face_windows.append(window)

        return concat_shapes(*face_windows)

    # Create windows for each face
    front_windows = create_face_windows('front')
    back_windows = create_face_windows('back')
    left_windows = create_face_windows('left')
    right_windows = create_face_windows('right')

    return concat_shapes(main_building, front_windows, back_windows, left_windows, right_windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float, color: tuple = (0.8, 0.7, 0.6)) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Roof
    roof_height = height * 0.5
    roof_color = (0.6, 0.3, 0.2)

    # Create a roof using a scaled and rotated cube
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=roof_color)

    # Position the roof on top of the house
    house_top = compute_shape_max(main_house)[1]
    roof = transform_shape(roof, translation_matrix((0, house_top + roof_height/2, 0)))

    # Add a door
    door_width = width * 0.2
    door_height = height * 0.4
    door_depth = 0.01
    door_color = (0.4, 0.2, 0.1)

    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=door_color)
    door = transform_shape(door, translation_matrix((0, -height/2 + door_height/2, depth/2)))

    return concat_shapes(main_house, roof, door)

@register("Creates a road segment")
def road(length: float, width: float = 0.5) -> Shape:
    road_color = (0.2, 0.2, 0.2)
    road_height = 0.01

    road_segment = primitive_call('cube', shape_kwargs={'scale': (width, road_height, length)}, color=road_color)

    # Add road markings
    marking_width = width * 0.1
    marking_length = length * 0.1
    marking_height = 0.015
    marking_color = (1.0, 1.0, 1.0)

    def marking_fn(i: int) -> Shape:
        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, marking_height, marking_length)}, color=marking_color)
        position = (0, road_height, -length/2 + marking_length/2 + i * marking_length * 2)
        return transform_shape(marking, translation_matrix(position))

    markings = loop(int(length / (marking_length * 2)), marking_fn)

    return concat_shapes(road_segment, markings)

@register("Creates a street lamp")
def street_lamp(height: float = 3.0) -> Shape:
    pole_radius = 0.05
    pole_color = (0.3, 0.3, 0.3)

    # Create pole
    pole = primitive_call('cylinder', shape_kwargs={'radius': pole_radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=pole_color)

    # Create lamp head
    lamp_radius = 0.15
    lamp_color = (0.9, 0.9, 0.6)

    lamp_head = primitive_call('sphere', shape_kwargs={'radius': lamp_radius}, color=lamp_color)
    lamp_head = transform_shape(lamp_head, translation_matrix((0, height, 0)))

    return concat_shapes(pole, lamp_head)

@register("Creates a tree")
def tree(height: float = 2.0) -> Shape:
    # Create trunk
    trunk_radius = 0.1
    trunk_height = height * 0.4
    trunk_color = (0.5, 0.3, 0.2)

    trunk = primitive_call('cylinder', shape_kwargs={'radius': trunk_radius, 'p0': (0, 0, 0), 'p1': (0, trunk_height, 0)}, color=trunk_color)

    # Create foliage
    foliage_radius = height * 0.3
    foliage_color = (0.1, 0.6, 0.1)

    foliage = primitive_call('sphere', shape_kwargs={'radius': foliage_radius}, color=foliage_color)
    foliage = transform_shape(foliage, translation_matrix((0, trunk_height + foliage_radius * 0.7, 0)))

    return concat_shapes(trunk, foliage)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, max_buildings: int = 6) -> Shape:
    buildings_list = []

    # Divide the block into a grid
    grid_size = int(math.sqrt(max_buildings))
    cell_width = width / grid_size
    cell_depth = depth / grid_size

    def create_building(i: int, j: int) -> Shape:
        # Randomize building properties
        building_type = np.random.choice(['skyscraper', 'building', 'house'])
        building_width = cell_width * np.random.uniform(0.6, 0.9)
        building_depth = cell_depth * np.random.uniform(0.6, 0.9)

        if building_type == 'skyscraper':
            building_height = np.random.uniform(4.0, 8.0)
            color = (np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7))
            building = library_call('skyscraper', width=building_width, height=building_height, depth=building_depth, color=color)
        elif building_type == 'building':
            building_height = np.random.uniform(1.5, 3.0)
            color = (np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8))
            building = library_call('building', width=building_width, height=building_height, depth=building_depth, color=color)
        else:
            building_height = np.random.uniform(0.8, 1.5)
            color = (np.random.uniform(0.7, 0.9), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7))
            building = library_call('house', width=building_width, height=building_height, depth=building_depth, color=color)

        # Position the building in the grid
        x_pos = -width/2 + cell_width/2 + i * cell_width
        z_pos = -depth/2 + cell_depth/2 + j * cell_depth

        # Get the bottom of the building to place it on the ground
        building_min = compute_shape_min(building)
        y_offset = -building_min[1]

        return transform_shape(building, translation_matrix((x_pos, y_offset, z_pos)))

    # Create buildings in a grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            # Randomly skip some positions to create variety
            if np.random.random() < 0.8:  # 80% chance to place a building
                buildings_list.append(create_building(i, j))

    return concat_shapes(*buildings_list)

@register("Creates a city district with blocks and roads")
def city_district(size: float, num_blocks: int = 3) -> Shape:
    district = []

    # Calculate block and road dimensions
    road_width = 1.0
    block_size = (size - (num_blocks + 1) * road_width) / num_blocks

    # Create city blocks
    for i in range(num_blocks):
        for j in range(num_blocks):
            x_pos = -size/2 + road_width + block_size/2 + i * (block_size + road_width)
            z_pos = -size/2 + road_width + block_size/2 + j * (block_size + road_width)

            block = library_call('city_block', width=block_size, depth=block_size)
            block = transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
            district.append(block)

    # Create horizontal roads
    for i in range(num_blocks + 1):
        z_pos = -size/2 + i * (block_size + road_width)
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, translation_matrix((0, 0, z_pos)))
        district.append(road)

    # Create vertical roads
    for i in range(num_blocks + 1):
        x_pos = -size/2 + i * (block_size + road_width)
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((x_pos, 0, 0)))
        district.append(road)

    # Add street lamps at intersections
    for i in range(num_blocks + 1):
        for j in range(num_blocks + 1):
            x_pos = -size/2 + i * (block_size + road_width)
            z_pos = -size/2 + j * (block_size + road_width)

            lamp = library_call('street_lamp')
            lamp = transform_shape(lamp, translation_matrix((x_pos + road_width/2, 0, z_pos + road_width/2)))
            district.append(lamp)

    # Add trees around the district
    num_trees = int(size * 0.8)

    def tree_fn(i: int) -> Shape:
        # Place trees around the perimeter
        side = i % 4
        pos_along_side = (i // 4) / (num_trees // 4)

        if side == 0:  # Top
            x_pos = -size/2 + pos_along_side * size
            z_pos = -size/2 - np.random.uniform(0.5, 1.5)
        elif side == 1:  # Right
            x_pos = size/2 + np.random.uniform(0.5, 1.5)
            z_pos = -size/2 + pos_along_side * size
        elif side == 2:  # Bottom
            x_pos = -size/2 + pos_along_side * size
            z_pos = size/2 + np.random.uniform(0.5, 1.5)
        else:  # Left
            x_pos = -size/2 - np.random.uniform(0.5, 1.5)
            z_pos = -size/2 + pos_along_side * size

        tree_height = np.random.uniform(1.5, 3.0)
        tree = library_call('tree', height=tree_height)
        return transform_shape(tree, translation_matrix((x_pos, 0, z_pos)))

    trees = loop(num_trees, tree_fn)
    district.append(trees)

    return concat_shapes(*district)

@register("Creates a complete city with multiple districts")
def city() -> Shape:
    # Create a ground plane
    ground_size = 100
    ground_height = 0.1
    ground_color = (0.3, 0.5, 0.3)

    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, ground_height, ground_size)}, color=ground_color)
    ground = transform_shape(ground, translation_matrix((0, -ground_height/2, 0)))

    # Create city districts
    districts = []

    # Central district with larger buildings
    central_district = library_call('city_district', size=30, num_blocks=4)
    districts.append(central_district)

    # Surrounding districts
    district_positions = [
        (35, 0, 0),    # East
        (-35, 0, 0),   # West
        (0, 0, 35),    # South
        (0, 0, -35),   # North
        (25, 0, 25),   # Southeast
        (-25, 0, 25),  # Southwest
        (25, 0, -25),  # Northeast
        (-25, 0, -25)  # Northwest
    ]

    for i, pos in enumerate(district_positions):
        district = library_call('city_district', size=20, num_blocks=3)
        district = transform_shape(district, translation_matrix(pos))
        districts.append(district)

    return concat_shapes(ground, *districts)
```

Provide your critiques and suggestions for improvement below in a formatted list.


![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/expert_00_refl_00_writer/renderings/exposed_city_rover_background_frame_00/rendering_traj_000.png)



![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/expert_00_refl_00_writer/renderings/exposed_city_rover_background_frame_00/rendering_traj_003.png)

