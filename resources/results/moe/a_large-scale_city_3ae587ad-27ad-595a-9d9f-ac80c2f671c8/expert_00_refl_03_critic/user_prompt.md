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
    # Create building centered at its base
    building_shape = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    # Move up so bottom is at y=0
    return transform_shape(building_shape, translation_matrix((0, height/2, 0)))

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building centered at its base
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    main_building = transform_shape(main_building, translation_matrix((0, height/2, 0)))

    # Window parameters
    window_width = width * 0.15
    window_height = window_width * 1.5
    window_depth = 0.01
    window_color = (0.9, 0.9, 1.0)

    # Calculate number of windows per side
    windows_per_width = max(2, int(width / (window_width * 1.5)))
    windows_per_height = max(3, int(height / (window_height * 1.5)))

    # Function to create windows on one face using loop
    def create_face_windows(face_direction: str) -> Shape:
        def window_fn(idx: int) -> Shape:
            i = idx % windows_per_width
            j = idx // windows_per_width

            if face_direction in ['front', 'back']:
                z_pos = depth/2 if face_direction == 'front' else -depth/2
                x_spacing = width / (windows_per_width + 1)
                y_spacing = height / (windows_per_height + 1)

                x_pos = ((i + 1) * x_spacing) - width/2
                y_pos = (j + 1) * y_spacing

                window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
                return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            else:  # 'left' or 'right'
                x_pos = width/2 if face_direction == 'right' else -width/2
                z_spacing = depth / (windows_per_width + 1)
                y_spacing = height / (windows_per_height + 1)

                z_pos = ((i + 1) * z_spacing) - depth/2
                y_pos = (j + 1) * y_spacing

                window = primitive_call('cube', shape_kwargs={'scale': (window_depth, window_height, window_width)}, color=window_color)
                return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

        return loop(windows_per_width * windows_per_height, window_fn)

    # Create windows for each face
    front_windows = create_face_windows('front')
    back_windows = create_face_windows('back')
    left_windows = create_face_windows('left')
    right_windows = create_face_windows('right')

    return concat_shapes(main_building, front_windows, back_windows, left_windows, right_windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float, color: tuple = (0.8, 0.7, 0.6)) -> Shape:
    # Main house centered at its base
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    main_house = transform_shape(main_house, translation_matrix((0, height/2, 0)))

    # Roof
    roof_height = height * 0.5
    roof_color = (0.6, 0.3, 0.2)

    # Create a roof using a scaled and rotated cube
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=roof_color)

    # Position the roof on top of the house
    roof = transform_shape(roof, translation_matrix((0, height + roof_height/2, 0)))

    # Add a door
    door_width = width * 0.2
    door_height = height * 0.4
    door_depth = 0.01
    door_color = (0.4, 0.2, 0.1)

    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=door_color)
    door = transform_shape(door, translation_matrix((0, door_height/2, depth/2)))

    return concat_shapes(main_house, roof, door)

@register("Creates a road segment with sidewalks")
def road(length: float, width: float = 1.0) -> Shape:
    road_color = (0.2, 0.2, 0.2)
    road_height = 0.02

    # Slightly elevate road to prevent z-fighting
    road_elevation = 0.01

    # Main road
    road_segment = primitive_call('cube', shape_kwargs={'scale': (width, road_height, length)}, color=road_color)
    road_segment = transform_shape(road_segment, translation_matrix((0, road_elevation, 0)))

    # Add road markings using loop
    marking_width = width * 0.05
    marking_length = length * 0.05
    marking_height = 0.025
    marking_color = (1.0, 1.0, 1.0)
    markings_count = int(length / (marking_length * 2))

    def marking_fn(i: int) -> Shape:
        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, marking_height, marking_length)}, color=marking_color)
        position = (0, road_elevation + road_height/2, -length/2 + marking_length/2 + i * marking_length * 2)
        return transform_shape(marking, translation_matrix(position))

    markings = loop(markings_count, marking_fn)

    # Add sidewalks
    sidewalk_width = width * 0.3
    sidewalk_height = 0.05
    sidewalk_color = (0.7, 0.7, 0.7)

    left_sidewalk = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, sidewalk_height, length)}, color=sidewalk_color)
    left_sidewalk = transform_shape(left_sidewalk, translation_matrix((-width/2 - sidewalk_width/2, sidewalk_height/2, 0)))

    right_sidewalk = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, sidewalk_height, length)}, color=sidewalk_color)
    right_sidewalk = transform_shape(right_sidewalk, translation_matrix((width/2 + sidewalk_width/2, sidewalk_height/2, 0)))

    return concat_shapes(road_segment, markings, left_sidewalk, right_sidewalk)

@register("Creates a street lamp")
def street_lamp(height: float = 2.5) -> Shape:
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

@register("Creates a traffic light")
def traffic_light(height: float = 3.0) -> Shape:
    pole_radius = 0.05
    pole_color = (0.3, 0.3, 0.3)

    # Create pole
    pole = primitive_call('cylinder', shape_kwargs={'radius': pole_radius, 'p0': (0, 0, 0), 'p1': (0, height, 0)}, color=pole_color)

    # Create traffic light box
    box_width = 0.2
    box_height = 0.6
    box_depth = 0.2
    box_color = (0.2, 0.2, 0.2)

    box = primitive_call('cube', shape_kwargs={'scale': (box_width, box_height, box_depth)}, color=box_color)
    box = transform_shape(box, translation_matrix((0, height - box_height/2, 0)))

    # Create lights
    light_radius = 0.06
    red_light = primitive_call('sphere', shape_kwargs={'radius': light_radius}, color=(1.0, 0.0, 0.0))
    yellow_light = primitive_call('sphere', shape_kwargs={'radius': light_radius}, color=(1.0, 1.0, 0.0))
    green_light = primitive_call('sphere', shape_kwargs={'radius': light_radius}, color=(0.0, 1.0, 0.0))

    red_light = transform_shape(red_light, translation_matrix((0, height - box_height/6, box_depth/2)))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, height - box_height/2, box_depth/2)))
    green_light = transform_shape(green_light, translation_matrix((0, height - 5*box_height/6, box_depth/2)))

    return concat_shapes(pole, box, red_light, yellow_light, green_light)

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

@register("Creates a park with trees and benches")
def park(width: float, depth: float) -> Shape:
    # Create grass base
    grass_height = 0.05
    grass_color = (0.2, 0.7, 0.2)

    grass = primitive_call('cube', shape_kwargs={'scale': (width, grass_height, depth)}, color=grass_color)
    grass = transform_shape(grass, translation_matrix((0, grass_height/2, 0)))

    # Add trees
    num_trees = int(width * depth / 10)

    def tree_fn(i: int) -> Shape:
        x = np.random.uniform(-width/2 + 1, width/2 - 1)
        z = np.random.uniform(-depth/2 + 1, depth/2 - 1)
        tree_height = np.random.uniform(1.5, 2.5)
        tree = library_call('tree', height=tree_height)
        return transform_shape(tree, translation_matrix((x, 0, z)))

    trees = loop(num_trees, tree_fn)

    # Add benches
    bench_width = 1.0
    bench_height = 0.4
    bench_depth = 0.4
    bench_color = (0.6, 0.4, 0.2)

    def bench_fn(i: int) -> Shape:
        # Place benches around the perimeter
        side = i % 4
        pos = (i // 4) / max(1, (num_benches // 4))

        if side == 0:  # Top
            x = -width/2 + width * pos
            z = -depth/2 + 1
        elif side == 1:  # Right
            x = width/2 - 1
            z = -depth/2 + depth * pos
        elif side == 2:  # Bottom
            x = -width/2 + width * pos
            z = depth/2 - 1
        else:  # Left
            x = -width/2 + 1
            z = -depth/2 + depth * pos

        bench = primitive_call('cube', shape_kwargs={'scale': (bench_width, bench_height, bench_depth)}, color=bench_color)
        bench = transform_shape(bench, translation_matrix((x, bench_height/2, z)))
        return bench

    num_benches = 8
    benches = loop(num_benches, bench_fn)

    return concat_shapes(grass, trees, benches)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, max_buildings: int = 6, block_type: str = 'mixed') -> Shape:
    # Set random seed for reproducibility
    np.random.seed(int(width * depth) % 1000)

    buildings_list = []

    # Divide the block into a grid
    grid_size = int(math.sqrt(max_buildings))
    cell_width = width / grid_size
    cell_depth = depth / grid_size

    def create_building(i: int, j: int) -> Shape:
        # Adjust building type probabilities based on block type
        if block_type == 'downtown':
            type_probs = [0.7, 0.3, 0.0]  # skyscraper, building, house
            height_range = (4.0, 8.0)
        elif block_type == 'residential':
            type_probs = [0.0, 0.3, 0.7]  # skyscraper, building, house
            height_range = (0.8, 2.0)
        elif block_type == 'commercial':
            type_probs = [0.2, 0.7, 0.1]  # skyscraper, building, house
            height_range = (1.5, 4.0)
        else:  # mixed
            type_probs = [0.3, 0.4, 0.3]  # skyscraper, building, house
            height_range = (1.0, 6.0)

        building_type = np.random.choice(['skyscraper', 'building', 'house'], p=type_probs)
        building_width = cell_width * np.random.uniform(0.6, 0.9)
        building_depth = cell_depth * np.random.uniform(0.6, 0.9)

        if building_type == 'skyscraper':
            building_height = np.random.uniform(height_range[0], height_range[1])
            color = (np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7))
            building = library_call('skyscraper', width=building_width, height=building_height, depth=building_depth, color=color)
        elif building_type == 'building':
            building_height = np.random.uniform(height_range[0] * 0.7, height_range[1] * 0.7)
            color = (np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8))
            building = library_call('building', width=building_width, height=building_height, depth=building_depth, color=color)
        else:
            building_height = np.random.uniform(height_range[0] * 0.5, height_range[1] * 0.5)
            color = (np.random.uniform(0.7, 0.9), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7))
            building = library_call('house', width=building_width, height=building_height, depth=building_depth, color=color)

        # Position the building in the grid
        x_pos = -width/2 + cell_width/2 + i * cell_width
        z_pos = -depth/2 + cell_depth/2 + j * cell_depth

        return transform_shape(building, translation_matrix((x_pos, 0, z_pos)))

    # Create buildings in a grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            # Randomly skip some positions to create variety
            if np.random.random() < 0.8:  # 80% chance to place a building
                buildings_list.append(create_building(i, j))

    # Add a small park in one corner if it's a residential or mixed block
    if block_type in ['residential', 'mixed'] and np.random.random() < 0.3:
        park_size = min(width, depth) * 0.3
        park_x = -width/2 + park_size/2
        park_z = -depth/2 + park_size/2
        park = library_call('park', width=park_size, depth=park_size)
        park = transform_shape(park, translation_matrix((park_x, 0, park_z)))
        buildings_list.append(park)

    return concat_shapes(*buildings_list)

@register("Creates a city district with blocks and roads")
def city_district(size: float, num_blocks: int = 3, district_type: str = 'mixed') -> Shape:
    # Set random seed for reproducibility
    np.random.seed(int(size * 100) % 1000)

    district = []

    # Calculate block and road dimensions
    road_width = 1.0
    block_size = (size - (num_blocks + 1) * road_width) / num_blocks

    # Create city blocks
    for i in range(num_blocks):
        for j in range(num_blocks):
            x_pos = -size/2 + road_width + block_size/2 + i * (block_size + road_width)
            z_pos = -size/2 + road_width + block_size/2 + j * (block_size + road_width)

            # Determine block type based on district type and position
            if district_type == 'downtown':
                block_type = 'downtown'
            elif district_type == 'residential':
                block_type = 'residential'
            elif district_type == 'commercial':
                block_type = 'commercial'
            else:  # mixed
                # Center blocks are more likely to be downtown
                distance_from_center = math.sqrt((i - num_blocks/2)**2 + (j - num_blocks/2)**2)
                if distance_from_center < num_blocks/4:
                    block_type = np.random.choice(['downtown', 'commercial', 'mixed'], p=[0.6, 0.3, 0.1])
                else:
                    block_type = np.random.choice(['residential', 'commercial', 'mixed'], p=[0.6, 0.3, 0.1])

            block = library_call('city_block', width=block_size, depth=block_size, block_type=block_type)
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

    # Add street lamps and traffic lights at intersections
    for i in range(num_blocks + 1):
        for j in range(num_blocks + 1):
            x_pos = -size/2 + i * (block_size + road_width)
            z_pos = -size/2 + j * (block_size + road_width)

            # Add street lamp at corner of intersection
            lamp = library_call('street_lamp')
            lamp = transform_shape(lamp, translation_matrix((x_pos - road_width/3, 0, z_pos - road_width/3)))
            district.append(lamp)

            # Add traffic light at major intersections
            if i > 0 and i < num_blocks and j > 0 and j < num_blocks:
                if (i + j) % 2 == 0:  # Only at some intersections
                    traffic_light_ns = library_call('traffic_light')
                    traffic_light_ns = transform_shape(traffic_light_ns, translation_matrix((x_pos - road_width/2, 0, z_pos)))

                    traffic_light_ew = library_call('traffic_light')
                    traffic_light_ew = transform_shape(traffic_light_ew, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
                    traffic_light_ew = transform_shape(traffic_light_ew, translation_matrix((x_pos, 0, z_pos - road_width/2)))

                    district.append(traffic_light_ns)
                    district.append(traffic_light_ew)

    # Add trees around the district
    num_trees = int(size * 0.8)

    def tree_fn(i: int) -> Shape:
        # Place trees around the perimeter with even distribution
        side = i % 4
        pos_along_side = (i // 4) / max(1, (num_trees // 4))

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
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a ground plane with subtle elevation
    ground_size = 100
    ground_height = 0.1
    ground_color = (0.3, 0.5, 0.3)

    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, ground_height, ground_size)}, color=ground_color)
    ground = transform_shape(ground, translation_matrix((0, -ground_height/2, 0)))

    # Create city districts
    districts = []

    # Central district with larger buildings (downtown)
    central_district = library_call('city_district', size=30, num_blocks=4, district_type='downtown')
    districts.append(central_district)

    # Surrounding districts with different types
    district_positions = [
        (35, 0, 0, 'commercial'),      # East
        (-35, 0, 0, 'residential'),    # West
        (0, 0, 35, 'commercial'),      # South
        (0, 0, -35, 'residential'),    # North
        (25, 0, 25, 'mixed'),          # Southeast
        (-25, 0, 25, 'residential'),   # Southwest
        (25, 0, -25, 'mixed'),         # Northeast
        (-25, 0, -25, 'residential')   # Northwest
    ]

    for pos in district_positions:
        district = library_call('city_district', size=20, num_blocks=3, district_type=pos[3])
        district = transform_shape(district, translation_matrix((pos[0], 0, pos[2])))
        districts.append(district)

    # Create a river through the city
    river_width = 5.0
    river_length = ground_size
    river_depth = 0.5
    river_color = (0.1, 0.4, 0.8)

    river = primitive_call('cube', shape_kwargs={'scale': (river_width, river_depth, river_length)}, color=river_color)
    river = transform_shape(river, translation_matrix((15, -ground_height/2, 0)))

    # Create bridges over the river
    bridge_width = 2.0
    bridge_height = 0.3
    bridge_length = river_width * 1.2
    bridge_color = (0.5, 0.5, 0.5)

    def bridge_fn(i: int) -> Shape:
        z_pos = -river_length/2 + river_length/(4+1) * (i+1)
        bridge = primitive_call('cube', shape_kwargs={'scale': (bridge_length, bridge_height, bridge_width)}, color=bridge_color)
        return transform_shape(bridge, translation_matrix((15, 0, z_pos)))

    bridges = loop(4, bridge_fn)

    return concat_shapes(ground, river, bridges, *districts)
```

Provide your critiques and suggestions for improvement below in a formatted list.


![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/expert_00_refl_02_writer/renderings/exposed_city_rover_background_frame_00/rendering_traj_000.png)



![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/expert_00_refl_02_writer/renderings/exposed_city_rover_background_frame_00/rendering_traj_003.png)

