Your task is to evaluate the following code proposals for the task described below and select the best one.

Task description:
a large-scale city

You will be presented with the following code proposals. Carefully examine the provided image rendered from the code proposal 1 to 4, horizontally concatenated in the same order of proposals. Non-compilable code proposals give a black image.
Please evaluate each based on:
1. Physical accuracy. No penetration or floating allowed unless desired by the task.
2. Aesthetics.

Code Proposal 1:
```python
from helper import *

"""
a large-scale city with improved positioning and details
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
    def create_face_windows(face_direction: str, density: float = 1.0) -> Shape:
        # Adjust window density based on face (some sides have fewer windows)
        actual_windows = int(windows_per_width * windows_per_height * density)
        if actual_windows == 0:
            return []

        def window_fn(idx: int) -> Shape:
            i = idx % windows_per_width
            j = idx // windows_per_width

            if j >= windows_per_height:
                j = j % windows_per_height

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

        return loop(actual_windows, window_fn)

    # Create windows for each face with varying density
    front_windows = create_face_windows('front', 1.0)
    back_windows = create_face_windows('back', 0.8)  # Fewer windows on back
    left_windows = create_face_windows('left', 0.9)
    right_windows = create_face_windows('right', 0.9)

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

    # Add windows
    window_width = width * 0.15
    window_height = window_width * 1.2
    window_depth = 0.01
    window_color = (0.9, 0.95, 1.0)

    # Front windows
    left_window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
    left_window = transform_shape(left_window, translation_matrix((-width/4, height/2, depth/2)))

    right_window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
    right_window = transform_shape(right_window, translation_matrix((width/4, height/2, depth/2)))

    return concat_shapes(main_house, roof, door, left_window, right_window)

@register("Creates a road segment with sidewalks and crosswalks")
def road(length: float, width: float = 1.0, add_crosswalk: bool = False) -> Shape:
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

    # Add edge markings
    edge_marking_width = width * 0.03

    left_edge = primitive_call('cube', shape_kwargs={'scale': (edge_marking_width, marking_height, length)}, color=marking_color)
    left_edge = transform_shape(left_edge, translation_matrix((-width/2 + edge_marking_width/2, road_elevation + road_height/2, 0)))

    right_edge = primitive_call('cube', shape_kwargs={'scale': (edge_marking_width, marking_height, length)}, color=marking_color)
    right_edge = transform_shape(right_edge, translation_matrix((width/2 - edge_marking_width/2, road_elevation + road_height/2, 0)))

    # Add sidewalks
    sidewalk_width = width * 0.3
    sidewalk_height = 0.05
    sidewalk_color = (0.7, 0.7, 0.7)

    left_sidewalk = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, sidewalk_height, length)}, color=sidewalk_color)
    left_sidewalk = transform_shape(left_sidewalk, translation_matrix((-width/2 - sidewalk_width/2, sidewalk_height/2, 0)))

    right_sidewalk = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, sidewalk_height, length)}, color=sidewalk_color)
    right_sidewalk = transform_shape(right_sidewalk, translation_matrix((width/2 + sidewalk_width/2, sidewalk_height/2, 0)))

    # Add crosswalk if requested
    if add_crosswalk:
        crosswalk_width = width
        crosswalk_length = width * 0.2
        crosswalk_stripes = 5

        def crosswalk_stripe_fn(i: int) -> Shape:
            stripe_width = crosswalk_width / (crosswalk_stripes * 2 - 1)
            stripe = primitive_call('cube', shape_kwargs={'scale': (stripe_width, marking_height, crosswalk_length)}, color=marking_color)
            x_pos = -crosswalk_width/2 + stripe_width/2 + i * stripe_width * 2
            return transform_shape(stripe, translation_matrix((x_pos, road_elevation + road_height/2, length/2 - crosswalk_length/2)))

        crosswalk = loop(crosswalk_stripes, crosswalk_stripe_fn)
        return concat_shapes(road_segment, markings, left_edge, right_edge, left_sidewalk, right_sidewalk, crosswalk)

    return concat_shapes(road_segment, markings, left_edge, right_edge, left_sidewalk, right_sidewalk)

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
def traffic_light(height: float = 3.0, orientation: float = 0.0) -> Shape:
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

    # Combine all parts
    traffic_light = concat_shapes(pole, box, red_light, yellow_light, green_light)

    # Apply orientation to face the correct direction
    if orientation != 0.0:
        traffic_light = transform_shape(traffic_light, rotation_matrix(orientation, (0, 1, 0), (0, 0, 0)))

    return traffic_light

@register("Creates a tree with specified type")
def tree(height: float = 2.0, tree_type: str = 'deciduous') -> Shape:
    # Create trunk
    trunk_radius = 0.1
    trunk_height = height * 0.4
    trunk_color = (0.5, 0.3, 0.2)

    trunk = primitive_call('cylinder', shape_kwargs={'radius': trunk_radius, 'p0': (0, 0, 0), 'p1': (0, trunk_height, 0)}, color=trunk_color)

    # Create foliage based on tree type
    if tree_type == 'pine':
        # Pine tree has a conical shape
        foliage_radius = height * 0.25
        foliage_height = height * 0.7
        foliage_color = (0.1, 0.4, 0.1)

        # Create a cone-like shape using a scaled cube
        foliage = primitive_call('cube', shape_kwargs={'scale': (foliage_radius*2, foliage_height, foliage_radius*2)}, color=foliage_color)
        foliage = transform_shape(foliage, translation_matrix((0, trunk_height + foliage_height/2, 0)))

    else:  # deciduous (round)
        foliage_radius = height * 0.3
        foliage_color = (0.1, 0.6, 0.1)

        foliage = primitive_call('sphere', shape_kwargs={'radius': foliage_radius}, color=foliage_color)
        foliage = transform_shape(foliage, translation_matrix((0, trunk_height + foliage_radius * 0.7, 0)))

    return concat_shapes(trunk, foliage)

@register("Creates a park with trees and benches")
def park(width: float, depth: float) -> Shape:
    # Create grass base with slight elevation variation
    grass_height = 0.05
    grass_color = (0.2, 0.7, 0.2)

    grass = primitive_call('cube', shape_kwargs={'scale': (width, grass_height, depth)}, color=grass_color)
    grass = transform_shape(grass, translation_matrix((0, grass_height/2, 0)))

    # Add trees with variety
    num_trees = int(width * depth / 10)

    def tree_fn(i: int) -> Shape:
        x = np.random.uniform(-width/2 + 1, width/2 - 1)
        z = np.random.uniform(-depth/2 + 1, depth/2 - 1)
        tree_height = np.random.uniform(1.5, 2.5)
        tree_type = 'deciduous' if np.random.random() < 0.7 else 'pine'
        tree = library_call('tree', height=tree_height, tree_type=tree_type)
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
            rotation = 0
        elif side == 1:  # Right
            x = width/2 - 1
            z = -depth/2 + depth * pos
            rotation = math.pi/2
        elif side == 2:  # Bottom
            x = -width/2 + width * pos
            z = depth/2 - 1
            rotation = math.pi
        else:  # Left
            x = -width/2 + 1
            z = -depth/2 + depth * pos
            rotation = 3*math.pi/2

        bench = primitive_call('cube', shape_kwargs={'scale': (bench_width, bench_height, bench_depth)}, color=bench_color)
        bench = transform_shape(bench, rotation_matrix(rotation, (0, 1, 0), (0, 0, 0)))
        bench = transform_shape(bench, translation_matrix((x, bench_height/2, z)))
        return bench

    num_benches = 8
    benches = loop(num_benches, bench_fn)

    # Add a small pond
    if width > 5 and depth > 5:
        pond_radius = min(width, depth) * 0.15
        pond_depth = 0.1
        pond_color = (0.1, 0.5, 0.8)

        pond = primitive_call('cylinder',
                             shape_kwargs={'radius': pond_radius,
                                          'p0': (0, -pond_depth, 0),
                                          'p1': (0, 0, 0)},
                             color=pond_color)

        # Position pond randomly in the park
        pond_x = np.random.uniform(-width/4, width/4)
        pond_z = np.random.uniform(-depth/4, depth/4)
        pond = transform_shape(pond, translation_matrix((pond_x, 0, pond_z)))

        return concat_shapes(grass, trees, benches, pond)

    return concat_shapes(grass, trees, benches)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, num_buildings: int = 6, block_type: str = 'mixed') -> Shape:
    buildings_list = []

    # Divide the block into a grid
    grid_size = int(math.sqrt(num_buildings))
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

            # Calculate number of buildings based on block size
            num_buildings = max(4, int((block_size * block_size) / 10))

            block = library_call('city_block', width=block_size, depth=block_size,
                                num_buildings=num_buildings, block_type=block_type)
            block = transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
            district.append(block)

    # Create horizontal roads
    for i in range(num_blocks + 1):
        z_pos = -size/2 + i * (block_size + road_width)
        # Add crosswalks at intersections
        road = library_call('road', length=size, width=road_width, add_crosswalk=(i > 0 and i < num_blocks))
        road = transform_shape(road, translation_matrix((0, 0, z_pos)))
        district.append(road)

    # Create vertical roads
    for i in range(num_blocks + 1):
        x_pos = -size/2 + i * (block_size + road_width)
        # Add crosswalks at intersections
        road = library_call('road', length=size, width=road_width, add_crosswalk=(i > 0 and i < num_blocks))
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((x_pos, 0, 0)))
        district.append(road)

    # Add street lamps along roads
    lamp_spacing = 5.0
    num_lamps_per_road = int(size / lamp_spacing)

    def street_lamp_fn(idx: int) -> Shape:
        road_idx = idx // num_lamps_per_road
        lamp_idx = idx % num_lamps_per_road

        if road_idx < num_blocks + 1:  # Horizontal roads
            x_pos = -size/2 + lamp_spacing/2 + lamp_idx * lamp_spacing
            z_pos = -size/2 + road_idx * (block_size + road_width)
            z_offset = road_width * 0.4  # Offset from road center

            lamp1 = library_call('street_lamp')
            lamp1 = transform_shape(lamp1, translation_matrix((x_pos, 0, z_pos - z_offset)))

            lamp2 = library_call('street_lamp')
            lamp2 = transform_shape(lamp2, translation_matrix((x_pos, 0, z_pos + z_offset)))

            return concat_shapes(lamp1, lamp2)
        else:  # Vertical roads
            vert_idx = road_idx - (num_blocks + 1)
            x_pos = -size/2 + vert_idx * (block_size + road_width)
            z_pos = -size/2 + lamp_spacing/2 + lamp_idx * lamp_spacing
            x_offset = road_width * 0.4  # Offset from road center

            lamp1 = library_call('street_lamp')
            lamp1 = transform_shape(lamp1, translation_matrix((x_pos - x_offset, 0, z_pos)))

            lamp2 = library_call('street_lamp')
            lamp2 = transform_shape(lamp2, translation_matrix((x_pos + x_offset, 0, z_pos)))

            return concat_shapes(lamp1, lamp2)

    # Only place lamps at a subset of possible positions to avoid too many objects
    num_lamp_positions = (num_blocks + 1) * 2 * num_lamps_per_road
    lamp_density = 0.3  # Only place 30% of possible lamps
    num_actual_lamps = int(num_lamp_positions * lamp_density)

    lamp_indices = np.random.choice(num_lamp_positions, num_actual_lamps, replace=False)
    for idx in lamp_indices:
        district.append(street_lamp_fn(idx))

    # Add traffic lights at intersections with proper orientation
    for i in range(1, num_blocks):
        for j in range(1, num_blocks):
            x_pos = -size/2 + i * (block_size + road_width)
            z_pos = -size/2 + j * (block_size + road_width)

            # Only add traffic lights at some intersections
            if (i + j) % 2 == 0:
                # Traffic light facing north
                traffic_light_n = library_call('traffic_light', orientation=0)
                traffic_light_n = transform_shape(traffic_light_n,
                                                translation_matrix((x_pos - road_width/2, 0, z_pos + road_width/2)))

                # Traffic light facing south
                traffic_light_s = library_call('traffic_light', orientation=math.pi)
                traffic_light_s = transform_shape(traffic_light_s,
                                                translation_matrix((x_pos + road_width/2, 0, z_pos - road_width/2)))

                # Traffic light facing east
                traffic_light_e = library_call('traffic_light', orientation=math.pi/2)
                traffic_light_e = transform_shape(traffic_light_e,
                                                translation_matrix((x_pos - road_width/2, 0, z_pos - road_width/2)))

                # Traffic light facing west
                traffic_light_w = library_call('traffic_light', orientation=-math.pi/2)
                traffic_light_w = transform_shape(traffic_light_w,
                                                translation_matrix((x_pos + road_width/2, 0, z_pos + road_width/2)))

                district.append(concat_shapes(traffic_light_n, traffic_light_s, traffic_light_e, traffic_light_w))

    return concat_shapes(*district)

@register("Creates a complete city with multiple districts")
def city() -> Shape:
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a ground plane with subtle elevation variations
    ground_size = 100
    ground_height = 0.1
    ground_color = (0.3, 0.5, 0.3)

    # Create base ground
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

    # Create a curved river through the city
    river_width = 5.0
    river_length = ground_size * 1.2
    river_depth = 0.5
    river_color = (0.1, 0.4, 0.8)

    # Create river segments to form a curve
    num_segments = 5
    segment_length = river_length / num_segments

    river_segments = []
    for i in range(num_segments):
        # Calculate curve parameters
        t = i / (num_segments - 1)
        x_pos = 15 + 10 * math.sin(t * math.pi)  # Curved path
        z_pos = -river_length/2 + i * segment_length
        angle = math.atan2(10 * math.pi * math.cos(t * math.pi) / river_length, 1)

        segment = primitive_call('cube', shape_kwargs={'scale': (river_width, river_depth, segment_length)}, color=river_color)
        segment = transform_shape(segment, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        segment = transform_shape(segment, translation_matrix((x_pos, -ground_height/2 - river_depth/2, z_pos)))
        river_segments.append(segment)

    river = concat_shapes(*river_segments)

    # Create bridges over the river
    bridge_width = 2.0
    bridge_height = 0.3
    bridge_length = river_width * 1.2
    bridge_color = (0.5, 0.5, 0.5)

    def bridge_fn(i: int) -> Shape:
        # Position bridges at regular intervals, adjusting for the river curve
        t = (i + 1) / 5
        x_pos = 15 + 10 * math.sin(t * math.pi)
        z_pos = -river_length/2 + river_length * t
        angle = math.atan2(10 * math.pi * math.cos(t * math.pi) / river_length, 1)

        bridge = primitive_call('cube', shape_kwargs={'scale': (bridge_length, bridge_height, bridge_width)}, color=bridge_color)
        bridge = transform_shape(bridge, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        bridge = transform_shape(bridge, translation_matrix((x_pos, -ground_height/2 + bridge_height/2, z_pos)))
        return bridge

    bridges = loop(4, bridge_fn)

    # Add a landmark building
    landmark_width = 5.0
    landmark_height = 15.0
    landmark_depth = 5.0
    landmark_color = (0.7, 0.7, 0.9)

    landmark = library_call('skyscraper', width=landmark_width, height=landmark_height, depth=landmark_depth, color=landmark_color)
    landmark = transform_shape(landmark, translation_matrix((0, 0, 0)))  # Center of the city

    # Add terrain variations
    num_terrain_bumps = 20

    def terrain_bump_fn(i: int) -> Shape:
        # Create random terrain bumps around the outskirts
        angle = np.random.uniform(0, 2 * math.pi)
        distance = np.random.uniform(40, 45)
        x_pos = distance * math.cos(angle)
        z_pos = distance * math.sin(angle)

        bump_width = np.random.uniform(3.0, 8.0)
        bump_height = np.random.uniform(0.2, 1.0)
        bump_depth = np.random.uniform(3.0, 8.0)
        bump_color = (0.25, 0.45, 0.25)  # Slightly darker than ground

        bump = primitive_call('cube', shape_kwargs={'scale': (bump_width, bump_height, bump_depth)}, color=bump_color)
        bump = transform_shape(bump, translation_matrix((x_pos, -ground_height/2 + bump_height/2, z_pos)))
        return bump

    terrain_bumps = loop(num_terrain_bumps, terrain_bump_fn)

    return concat_shapes(ground, river, bridges, landmark, terrain_bumps, *districts)
```

Code Proposal 2:
```python
from helper import *

"""
a large-scale city
"""

# Set random seed for reproducibility
np.random.seed(42)

@register("Creates a building with specified dimensions and color")
def building(width: float, height: float, depth: float, color: tuple = (0.7, 0.7, 0.7)) -> Shape:
    return primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Add windows with a reasonable limit
    window_width = width * 0.15
    window_height = height * 0.05
    window_depth = 0.01
    window_color = (0.9, 0.9, 0.7)

    # Calculate number of windows per side with limits to prevent excessive windows
    windows_per_row = min(max(int(width / (window_width * 1.5)), 2), 5)
    windows_per_column = min(max(int(height / (window_height * 1.5)), 4), 10)

    def window_loop_fn(i):
        row = i % windows_per_row
        col = (i // windows_per_row) % windows_per_column
        side = i // (windows_per_row * windows_per_column)

        if side < 4:  # 4 sides of the building
            window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)

            # Position for each side
            if side == 0:  # Front
                x_pos = (row - (windows_per_row - 1) / 2) * (width / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                z_pos = depth / 2
                # Apply translation first, then rotation to ensure windows are flush with surface
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 1:  # Back
                x_pos = (row - (windows_per_row - 1) / 2) * (width / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                z_pos = -depth / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
                window = transform_shape(window, rotation_matrix(math.pi, (0, 1, 0), (x_pos, y_pos, z_pos)))
            elif side == 2:  # Left
                z_pos = (row - (windows_per_row - 1) / 2) * (depth / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                x_pos = -width / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
                window = transform_shape(window, rotation_matrix(-math.pi/2, (0, 1, 0), (x_pos, y_pos, z_pos)))
            elif side == 3:  # Right
                z_pos = (row - (windows_per_row - 1) / 2) * (depth / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                x_pos = width / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
                window = transform_shape(window, rotation_matrix(math.pi/2, (0, 1, 0), (x_pos, y_pos, z_pos)))

            return window
        return []

    total_windows = windows_per_row * windows_per_column * 4
    windows = loop(total_windows, window_loop_fn)

    # Add roof structure
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 0.5, height * 0.1, depth * 0.5)}, color=(0.4, 0.4, 0.5))
    roof = transform_shape(roof, translation_matrix((0, height / 2 + height * 0.05, 0)))

    antenna = primitive_call('cylinder', shape_kwargs={'radius': width * 0.02, 'p0': (0, height / 2 + height * 0.1, 0), 'p1': (0, height / 2 + height * 0.3, 0)}, color=(0.3, 0.3, 0.3))

    return concat_shapes(main_building, windows, roof, antenna)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float, color: tuple = (0.8, 0.7, 0.6)) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

    # Improved roof implementation
    roof_height = height * 0.4
    roof_color = (0.6, 0.3, 0.2)

    # Create a triangular roof using two rectangular prisms
    roof_left = primitive_call('cube', shape_kwargs={'scale': (width, roof_height, depth)}, color=roof_color)
    roof_right = primitive_call('cube', shape_kwargs={'scale': (width, roof_height, depth)}, color=roof_color)

    # Position and rotate the roof parts
    roof_left = transform_shape(roof_left, rotation_matrix(math.pi/4, (0, 0, 1), (0, 0, 0)))
    roof_left = transform_shape(roof_left, translation_matrix((-width/4, height/2 + roof_height/2, 0)))

    roof_right = transform_shape(roof_right, rotation_matrix(-math.pi/4, (0, 0, 1), (0, 0, 0)))
    roof_right = transform_shape(roof_right, translation_matrix((width/4, height/2 + roof_height/2, 0)))

    # Door
    door_width = width * 0.3
    door_height = height * 0.6
    door_depth = 0.01
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=(0.4, 0.2, 0.1))
    door = transform_shape(door, translation_matrix((0, -height/2 + door_height/2, depth/2 + door_depth/2)))

    # Windows
    window_width = width * 0.2
    window_height = height * 0.2
    window_depth = 0.01
    window_color = (0.9, 0.9, 1.0)

    window1 = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
    window1 = transform_shape(window1, translation_matrix((-width/4, 0, depth/2 + window_depth/2)))

    window2 = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
    window2 = transform_shape(window2, translation_matrix((width/4, 0, depth/2 + window_depth/2)))

    return concat_shapes(main_house, roof_left, roof_right, door, window1, window2)

@register("Creates a road segment with realistic markings")
def road(length: float, width: float = 1.0, color: tuple = (0.2, 0.2, 0.2)) -> Shape:
    road_height = 0.05
    road_shape = primitive_call('cube', shape_kwargs={'scale': (width, road_height, length)}, color=color)

    # Add road markings with more realistic spacing
    marking_width = width * 0.05
    marking_length = length * 0.05
    marking_color = (1.0, 1.0, 1.0)
    marking_gap = length * 0.07  # Gap between markings

    def marking_loop_fn(i):
        # Calculate position with gaps between markings
        z_pos = (i - 4.5) * (marking_length + marking_gap)

        # Add some randomness to make it more realistic
        z_pos += np.random.uniform(-length * 0.01, length * 0.01)

        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, road_height * 1.01, marking_length)}, color=marking_color)
        return transform_shape(marking, translation_matrix((0, 0, z_pos)))

    markings = loop(10, marking_loop_fn)

    # Add stop line at one end of the road
    stop_line = primitive_call('cube', shape_kwargs={'scale': (width * 0.8, road_height * 1.01, width * 0.1)}, color=marking_color)
    stop_line = transform_shape(stop_line, translation_matrix((0, 0, length/2 - width * 0.1)))

    return concat_shapes(road_shape, markings, stop_line)

@register("Creates a park with trees and paths")
def park(width: float, depth: float) -> Shape:
    # Ground with texture variation
    ground_base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))

    # Add texture variation with small patches of different green shades
    def grass_patch_fn(i):
        patch_size = min(width, depth) * 0.15
        x_pos = np.random.uniform(-width/2 + patch_size, width/2 - patch_size)
        z_pos = np.random.uniform(-depth/2 + patch_size, depth/2 - patch_size)

        # Random shade of green
        color = (0.1, np.random.uniform(0.5, 0.7), 0.1)

        patch = primitive_call('cube', shape_kwargs={'scale': (patch_size, 0.11, patch_size)}, color=color)
        return transform_shape(patch, translation_matrix((x_pos, 0, z_pos)))

    grass_patches = loop(8, grass_patch_fn)
    ground = concat_shapes(ground_base, grass_patches)

    # Add paths
    path_width = width * 0.15
    path_color = (0.8, 0.7, 0.6)

    # Horizontal path
    h_path = primitive_call('cube', shape_kwargs={'scale': (width, 0.12, path_width)}, color=path_color)

    # Vertical path
    v_path = primitive_call('cube', shape_kwargs={'scale': (path_width, 0.12, depth)}, color=path_color)

    # Add a central circular area
    center_radius = min(width, depth) * 0.15
    center_area = primitive_call('cylinder',
                                shape_kwargs={'radius': center_radius,
                                             'p0': (0, 0, 0),
                                             'p1': (0, 0.12, 0)},
                                color=path_color)
    center_area = transform_shape(center_area, rotation_matrix(math.pi/2, (1, 0, 0), (0, 0, 0)))

    # Add a fountain in the center
    fountain_base = primitive_call('cylinder',
                                  shape_kwargs={'radius': center_radius * 0.7,
                                               'p0': (0, 0.12, 0),
                                               'p1': (0, 0.2, 0)},
                                  color=(0.7, 0.7, 0.7))
    fountain_water = primitive_call('cylinder',
                                   shape_kwargs={'radius': center_radius * 0.5,
                                                'p0': (0, 0.2, 0),
                                                'p1': (0, 0.25, 0)},
                                   color=(0.6, 0.8, 0.9))
    fountain = concat_shapes(fountain_base, fountain_water)

    def tree_loop_fn(i):
        # Calculate position
        row = i % 5
        col = i // 5
        x_pos = (row - 2) * (width / 5)
        z_pos = (col - 2) * (depth / 5)

        # Add some randomness to positions
        x_pos += np.random.uniform(-width/12, width/12)
        z_pos += np.random.uniform(-depth/12, depth/12)

        # Skip trees that would be on the paths
        if abs(x_pos) < path_width/2 or abs(z_pos) < path_width/2:
            return []

        # Skip trees that would be in the central area
        if (x_pos**2 + z_pos**2) < center_radius**2:
            return []

        # Create different types of trees
        tree_type = i % 3  # 3 different tree types

        # Create tree trunk - ensure it starts at the ground level
        trunk_height = np.random.uniform(0.3, 0.6)
        trunk_radius = np.random.uniform(0.05, 0.1)
        trunk = primitive_call('cylinder',
                              shape_kwargs={'radius': trunk_radius,
                                           'p0': (x_pos, 0.1, z_pos),  # Start at ground level
                                           'p1': (x_pos, 0.1 + trunk_height, z_pos)},
                              color=(0.5, 0.3, 0.1))

        # Create tree foliage based on type
        if tree_type == 0:
            # Round tree
            foliage_radius = np.random.uniform(0.3, 0.5)
            foliage = primitive_call('sphere',
                                    shape_kwargs={'radius': foliage_radius},
                                    color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
            foliage = transform_shape(foliage, translation_matrix((x_pos, 0.1 + trunk_height + foliage_radius * 0.7, z_pos)))
            tree = concat_shapes(trunk, foliage)
        elif tree_type == 1:
            # Conical tree
            foliage_width = np.random.uniform(0.3, 0.5)
            foliage_height = np.random.uniform(0.6, 0.9)
            foliage = primitive_call('cube',
                                    shape_kwargs={'scale': (foliage_width, foliage_height, foliage_width)},
                                    color=(0.0, np.random.uniform(0.4, 0.7), 0.0))
            # Transform to make it conical
            foliage = transform_shape(foliage, scale_matrix(1.0, (x_pos, 0.1 + trunk_height + foliage_height/2, z_pos)))
            foliage = transform_shape(foliage, translation_matrix((x_pos, 0.1 + trunk_height + foliage_height/2, z_pos)))
            tree = concat_shapes(trunk, foliage)
        else:
            # Bushy tree (multiple spheres)
            foliage_radius = np.random.uniform(0.2, 0.4)
            foliage1 = primitive_call('sphere',
                                     shape_kwargs={'radius': foliage_radius},
                                     color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
            foliage1 = transform_shape(foliage1, translation_matrix((x_pos, 0.1 + trunk_height + foliage_radius, z_pos)))

            foliage2 = primitive_call('sphere',
                                     shape_kwargs={'radius': foliage_radius * 0.8},
                                     color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
            foliage2 = transform_shape(foliage2, translation_matrix((x_pos + foliage_radius*0.5, 0.1 + trunk_height + foliage_radius*0.7, z_pos)))

            foliage3 = primitive_call('sphere',
                                     shape_kwargs={'radius': foliage_radius * 0.8},
                                     color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
            foliage3 = transform_shape(foliage3, translation_matrix((x_pos - foliage_radius*0.5, 0.1 + trunk_height + foliage_radius*0.7, z_pos)))

            tree = concat_shapes(trunk, foliage1, foliage2, foliage3)

        # Add a bench near some trees (randomly)
        if np.random.random() < 0.2:  # 20% chance of adding a bench
            bench_seat = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.05, 0.2)}, color=(0.6, 0.4, 0.2))
            bench_back = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.2, 0.05)}, color=(0.6, 0.4, 0.2))

            bench_seat = transform_shape(bench_seat, translation_matrix((x_pos + 0.3, 0.15, z_pos)))
            bench_back = transform_shape(bench_back, translation_matrix((x_pos + 0.3, 0.25, z_pos - 0.075)))

            return concat_shapes(tree, bench_seat, bench_back)

        return tree

    trees = loop(25, tree_loop_fn)

    return concat_shapes(ground, h_path, v_path, center_area, fountain, trees)

@register("Creates a traffic light with proper orientation")
def traffic_light(orientation: int = 0) -> Shape:
    # Pole
    pole = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.05,
                                      'p0': (0, 0, 0),
                                      'p1': (0, 1.5, 0)},
                         color=(0.3, 0.3, 0.3))

    # Light housing
    housing = primitive_call('cube', shape_kwargs={'scale': (0.2, 0.5, 0.2)}, color=(0.1, 0.1, 0.1))
    housing = transform_shape(housing, translation_matrix((0, 1.5, 0)))

    # Lights
    red_light = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(1.0, 0.0, 0.0))
    red_light = transform_shape(red_light, translation_matrix((0, 1.65, 0.1)))

    yellow_light = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(1.0, 1.0, 0.0))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, 1.5, 0.1)))

    green_light = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(0.0, 1.0, 0.0))
    green_light = transform_shape(green_light, translation_matrix((0, 1.35, 0.1)))

    traffic_light = concat_shapes(pole, housing, red_light, yellow_light, green_light)

    # Rotate based on orientation (0=default, 1=90°, 2=180°, 3=270°)
    if orientation > 0:
        traffic_light = transform_shape(traffic_light,
                                       rotation_matrix(orientation * math.pi/2, (0, 1, 0), (0, 0, 0)))

    return traffic_light

@register("Creates a street lamp")
def street_lamp() -> Shape:
    # Pole
    pole = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.03,
                                      'p0': (0, 0, 0),
                                      'p1': (0, 2.0, 0)},
                         color=(0.3, 0.3, 0.3))

    # Lamp head
    lamp_head = primitive_call('sphere',
                              shape_kwargs={'radius': 0.1},
                              color=(0.8, 0.8, 0.6))
    lamp_head = transform_shape(lamp_head, translation_matrix((0, 2.0, 0.1)))

    # Lamp support
    lamp_support = primitive_call('cylinder',
                                 shape_kwargs={'radius': 0.02,
                                              'p0': (0, 1.9, 0),
                                              'p1': (0, 2.0, 0.1)},
                                 color=(0.3, 0.3, 0.3))

    return concat_shapes(pole, lamp_head, lamp_support)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, is_downtown: bool = False) -> Shape:
    # Create the base with slight elevation variation
    base_height = 0.1 + np.random.uniform(0, 0.05)  # Add slight random height
    base = primitive_call('cube', shape_kwargs={'scale': (width, base_height, depth)}, color=(0.6, 0.6, 0.6))

    # Create a grid for building placement to prevent overlap
    grid_size = 3
    cell_width = width / grid_size
    cell_depth = depth / grid_size

    def building_loop_fn(i):
        # Calculate position using grid
        row = i % grid_size
        col = i // grid_size

        # Calculate center of the grid cell
        x_pos = (row - (grid_size-1)/2) * cell_width
        z_pos = (col - (grid_size-1)/2) * cell_depth

        # Add small randomness within the cell to avoid perfect alignment
        x_pos += np.random.uniform(-cell_width * 0.1, cell_width * 0.1)
        z_pos += np.random.uniform(-cell_depth * 0.1, cell_depth * 0.1)

        # Determine building size to fit within the cell
        building_width = cell_width * np.random.uniform(0.6, 0.8)
        building_depth = cell_depth * np.random.uniform(0.6, 0.8)

        # Adjust height based on whether it's downtown (taller) or suburbs
        if is_downtown:
            building_height = np.random.uniform(2.0, 5.0)
            # Higher chance of skyscrapers downtown
            building_type_probs = [0.7, 0.25, 0.05]  # [skyscraper, building, house]
        else:
            building_height = np.random.uniform(1.0, 2.5)
            # Higher chance of houses in suburbs
            building_type_probs = [0.1, 0.4, 0.5]  # [skyscraper, building, house]

        # Randomly choose between different building types based on probabilities
        building_type = np.random.choice(['skyscraper', 'building', 'house'], p=building_type_probs)

        # Add architectural variety with different colors
        if building_type == 'skyscraper':
            # More variety in skyscraper styles
            style = np.random.randint(0, 3)
            if style == 0:  # Glass skyscraper
                color = (np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7), np.random.uniform(0.7, 0.9))
            elif style == 1:  # Stone skyscraper
                color = (np.random.uniform(0.5, 0.7), np.random.uniform(0.5, 0.7), np.random.uniform(0.5, 0.7))
            else:  # Modern skyscraper
                color = (np.random.uniform(0.3, 0.5), np.random.uniform(0.3, 0.5), np.random.uniform(0.4, 0.6))

            building_shape = library_call('skyscraper',
                                         width=building_width,
                                         height=building_height * 2,
                                         depth=building_depth,
                                         color=color)
        elif building_type == 'building':
            building_shape = library_call('building',
                                         width=building_width,
                                         height=building_height,
                                         depth=building_depth,
                                         color=(np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8)))
        else:
            building_shape = library_call('house',
                                         width=building_width,
                                         height=building_height * 0.7,
                                         depth=building_depth,
                                         color=(np.random.uniform(0.7, 0.9), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7)))

        # Position the building on top of the base
        building_bottom = compute_shape_min(building_shape)[1]
        base_top = compute_shape_max(base)[1]
        building_shape = transform_shape(building_shape, translation_matrix((x_pos, base_top - building_bottom, z_pos)))

        return building_shape

    buildings = loop(grid_size * grid_size, building_loop_fn)

    # Add street lamps around the perimeter
    def lamp_loop_fn(i):
        if i < 4:  # Four corners
            x_pos = width/2 - 0.3 if i % 2 == 0 else -width/2 + 0.3
            z_pos = depth/2 - 0.3 if i < 2 else -depth/2 + 0.3
            lamp = library_call('street_lamp')
            return transform_shape(lamp, translation_matrix((x_pos, base_height, z_pos)))
        return []

    lamps = loop(4, lamp_loop_fn)

    return concat_shapes(base, buildings, lamps)

@register("Creates a city intersection with traffic lights")
def intersection(road_width: float = 1.0) -> Shape:
    # Create the intersection base
    intersection_base = primitive_call('cube',
                                      shape_kwargs={'scale': (road_width * 2, 0.05, road_width * 2)},
                                      color=(0.2, 0.2, 0.2))

    # Add traffic lights at the corners with proper orientation
    traffic_light1 = library_call('traffic_light', orientation=2)  # Facing north
    traffic_light1 = transform_shape(traffic_light1,
                                    translation_matrix((road_width * 0.8, 0, road_width * 0.8)))

    traffic_light2 = library_call('traffic_light', orientation=3)  # Facing east
    traffic_light2 = transform_shape(traffic_light2,
                                    translation_matrix((-road_width * 0.8, 0, road_width * 0.8)))

    traffic_light3 = library_call('traffic_light', orientation=0)  # Facing south
    traffic_light3 = transform_shape(traffic_light3,
                                    translation_matrix((road_width * 0.8, 0, -road_width * 0.8)))

    traffic_light4 = library_call('traffic_light', orientation=1)  # Facing west
    traffic_light4 = transform_shape(traffic_light4,
                                    translation_matrix((-road_width * 0.8, 0, -road_width * 0.8)))

    # Add crosswalk markings
    crosswalk_color = (1.0, 1.0, 1.0)

    def crosswalk_marking_fn(i):
        # Create horizontal crosswalk markings
        if i < 5:
            marking = primitive_call('cube',
                                    shape_kwargs={'scale': (0.2, 0.06, road_width * 0.1)},
                                    color=crosswalk_color)
            x_pos = (i - 2) * 0.25
            return transform_shape(marking, translation_matrix((x_pos, 0, road_width * 0.6)))
        # Create vertical crosswalk markings
        else:
            marking = primitive_call('cube',
                                    shape_kwargs={'scale': (road_width * 0.1, 0.06, 0.2)},
                                    color=crosswalk_color)
            z_pos = ((i - 5) - 2) * 0.25
            return transform_shape(marking, translation_matrix((road_width * 0.6, 0, z_pos)))

    crosswalk_markings = loop(10, crosswalk_marking_fn)

    # Add stop lines at each entrance to the intersection
    stop_line1 = primitive_call('cube', shape_kwargs={'scale': (road_width * 0.8, 0.06, road_width * 0.1)}, color=crosswalk_color)
    stop_line1 = transform_shape(stop_line1, translation_matrix((0, 0, road_width * 0.9)))

    stop_line2 = primitive_call('cube', shape_kwargs={'scale': (road_width * 0.8, 0.06, road_width * 0.1)}, color=crosswalk_color)
    stop_line2 = transform_shape(stop_line2, translation_matrix((0, 0, -road_width * 0.9)))

    stop_line3 = primitive_call('cube', shape_kwargs={'scale': (road_width * 0.1, 0.06, road_width * 0.8)}, color=crosswalk_color)
    stop_line3 = transform_shape(stop_line3, translation_matrix((road_width * 0.9, 0, 0)))

    stop_line4 = primitive_call('cube', shape_kwargs={'scale': (road_width * 0.1, 0.06, road_width * 0.8)}, color=crosswalk_color)
    stop_line4 = transform_shape(stop_line4, translation_matrix((-road_width * 0.9, 0, 0)))

    return concat_shapes(intersection_base, traffic_light1, traffic_light2,
                         traffic_light3, traffic_light4, crosswalk_markings,
                         stop_line1, stop_line2, stop_line3, stop_line4)

@register("Creates a city grid with blocks and roads")
def city_grid(size: int = 5, block_size: float = 4.0, road_width: float = 1.0, is_downtown: bool = False) -> Shape:
    city = []

    # Calculate total unit size (block + road)
    total_unit_size = block_size + road_width

    # Create city blocks with slight terrain variation
    def block_loop_fn(i):
        row = i % size
        col = i // size

        # Calculate position with roads in between
        x_pos = (row - (size-1)/2) * total_unit_size
        z_pos = (col - (size-1)/2) * total_unit_size

        # Add slight elevation variation
        y_pos = np.random.uniform(-0.05, 0.05) if not is_downtown else 0

        # Randomly choose between regular block and park
        if is_downtown:
            park_chance = 0.1  # Less parks downtown
        else:
            park_chance = 0.2  # More parks in suburbs

        if np.random.random() < park_chance:
            block = library_call('park', width=block_size, depth=block_size)
        else:
            block = library_call('city_block', width=block_size, depth=block_size, is_downtown=is_downtown)

        return transform_shape(block, translation_matrix((x_pos, y_pos, z_pos)))

    blocks = loop(size * size, block_loop_fn)
    city.append(blocks)

    # Create horizontal roads
    def h_road_loop_fn(i):
        row = i % size
        col = i // size

        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size
        z_pos = (col - (size-1)/2) * total_unit_size + block_size/2 + road_width/2

        road = library_call('road', length=block_size, width=road_width)
        return transform_shape(road, translation_matrix((x_pos, 0, z_pos)))

    h_roads = loop(size * (size-1), h_road_loop_fn)
    city.append(h_roads)

    # Create vertical roads
    def v_road_loop_fn(i):
        row = i % (size-1)
        col = i // (size-1)

        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size + block_size/2 + road_width/2
        z_pos = (col - (size-1)/2) * total_unit_size

        road = library_call('road', length=block_size, width=road_width)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        return transform_shape(road, translation_matrix((x_pos, 0, z_pos)))

    v_roads = loop((size-1) * size, v_road_loop_fn)
    city.append(v_roads)

    # Create intersections
    def intersection_loop_fn(i):
        row = i % (size-1)
        col = i // (size-1)

        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size + block_size/2 + road_width/2
        z_pos = (col - (size-1)/2) * total_unit_size + block_size/2 + road_width/2

        intersection_shape = library_call('intersection', road_width=road_width)
        return transform_shape(intersection_shape, translation_matrix((x_pos, 0, z_pos)))

    intersections = loop((size-1) * (size-1), intersection_loop_fn)
    city.append(intersections)

    # Add street lamps along roads
    def street_lamp_fn(i):
        if i < size * 2:  # Lamps along horizontal roads
            row = i % size
            col = i // size

            x_pos = (row - (size-1)/2) * total_unit_size
            z_pos = (col - (size-1)/2) * total_unit_size + block_size/2 + road_width/2
            z_pos += (1 if col % 2 == 0 else -1) * road_width * 0.3  # Offset to side of road

            lamp = library_call('street_lamp')
            return transform_shape(lamp, translation_matrix((x_pos, 0, z_pos)))
        else:  # Lamps along vertical roads
            idx = i - size * 2
            row = idx % (size-1)
            col = idx // (size-1)

            x_pos = (row - (size-1)/2) * total_unit_size + block_size/2 + road_width/2
            z_pos = (col - (size-1)/2) * total_unit_size
            x_pos += (1 if row % 2 == 0 else -1) * road_width * 0.3  # Offset to side of road

            lamp = library_call('street_lamp')
            return transform_shape(lamp, translation_matrix((x_pos, 0, z_pos)))

    # Add a reasonable number of street lamps
    street_lamps = loop(size * 2 + (size-1) * 2, street_lamp_fn)
    city.append(street_lamps)

    return concat_shapes(*city)

@register("Creates a parking lot")
def parking_lot(width: float, depth: float) -> Shape:
    # Base
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.05, depth)}, color=(0.3, 0.3, 0.3))

    # Parking spaces
    space_width = 0.8
    space_depth = 1.6
    spacing = 0.1

    # Calculate number of spaces per row and column
    rows_per_side = int((depth - 2) / (space_depth + spacing))
    spaces_per_row = int((width - 2) / (space_width + spacing))

    def parking_space_fn(i):
        row = i % rows_per_side
        col = (i // rows_per_side) % spaces_per_row
        side = i // (rows_per_side * spaces_per_row)

        if side < 2:  # Two sides of parking spaces
            # Create parking space
            space = primitive_call('cube', shape_kwargs={'scale': (space_width, 0.06, space_depth)}, color=(0.25, 0.25, 0.25))

            # Add parking line markings
            line1 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.07, space_depth)}, color=(1.0, 1.0, 1.0))
            line2 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.07, space_depth)}, color=(1.0, 1.0, 1.0))

            line1 = transform_shape(line1, translation_matrix((space_width/2, 0, 0)))
            line2 = transform_shape(line2, translation_matrix((-space_width/2, 0, 0)))

            space_with_lines = concat_shapes(space, line1, line2)

            # Position based on side
            if side == 0:  # Left side
                x_pos = -width/2 + 1 + space_width/2 + col * (space_width + spacing)
                z_pos = -depth/2 + 1 + space_depth/2 + row * (space_depth + spacing)
            else:  # Right side
                x_pos = -width/2 + 1 + space_width/2 + col * (space_width + spacing)
                z_pos = depth/2 - 1 - space_depth/2 - row * (space_depth + spacing)

            return transform_shape(space_with_lines, translation_matrix((x_pos, 0, z_pos)))
        return []

    # Calculate total number of parking spaces
    total_spaces = rows_per_side * spaces_per_row * 2

    parking_spaces = loop(total_spaces, parking_space_fn)

    # Central driving lane
    lane = primitive_call('cube', shape_kwargs={'scale': (width - 2, 0.06, depth - 2 - 2*space_depth - 2*spacing)}, color=(0.2, 0.2, 0.2))

    # Add entrance/exit
    entrance = primitive_call('cube', shape_kwargs={'scale': (2.0, 0.06, 2.0)}, color=(0.2, 0.2, 0.2))
    entrance = transform_shape(entrance, translation_matrix((0, 0, -depth/2 - 1.0)))

    # Add parking lot markings
    def marking_fn(i):
        if i < 2:  # Entrance arrows
            arrow = primitive_call('cube', shape_kwargs={'scale': (0.5, 0.07, 0.1)}, color=(1.0, 1.0, 1.0))
            arrow = transform_shape(arrow, translation_matrix((i - 0.5, 0, -depth/2 - 0.5)))
            return arrow
        return []

    markings = loop(2, marking_fn)

    return concat_shapes(base, parking_spaces, lane, entrance, markings)

@register("Creates a water feature")
def water_feature(width: float, depth: float) -> Shape:
    # Water body
    water = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.1, 0.4, 0.8))

    # Shore/edge
    shore = primitive_call('cube', shape_kwargs={'scale': (width + 0.2, 0.08, depth + 0.2)}, color=(0.8, 0.7, 0.6))
    shore = transform_shape(shore, translation_matrix((0, -0.01, 0)))

    # Add some texture to the water
    def wave_fn(i):
        x_pos = np.random.uniform(-width/2 + 0.2, width/2 - 0.2)
        z_pos = np.random.uniform(-depth/2 + 0.2, depth/2 - 0.2)
        wave_size = np.random.uniform(0.1, 0.3)

        wave = primitive_call('cube', shape_kwargs={'scale': (wave_size, 0.01, wave_size)}, color=(0.2, 0.5, 0.9))
        return transform_shape(wave, translation_matrix((x_pos, 0.05, z_pos)))

    waves = loop(10, wave_fn)

    return concat_shapes(shore, water, waves)

@register("Creates a landmark building")
def landmark_building() -> Shape:
    # Base
    base_width = 3.0
    base_height = 1.0
    base_depth = 3.0
    base = primitive_call('cube', shape_kwargs={'scale': (base_width, base_height, base_depth)}, color=(0.7, 0.7, 0.7))

    # Middle section
    middle_width = 2.0
    middle_height = 3.0
    middle_depth = 2.0
    middle = primitive_call('cube', shape_kwargs={'scale': (middle_width, middle_height, middle_depth)}, color=(0.6, 0.6, 0.7))
    middle = transform_shape(middle, translation_matrix((0, base_height/2 + middle_height/2, 0)))

    # Top section - dome
    dome_radius = 1.0
    dome = primitive_call('sphere', shape_kwargs={'radius': dome_radius}, color=(0.8, 0.8, 0.9))
    dome = transform_shape(dome, translation_matrix((0, base_height/2 + middle_height + dome_radius*0.5, 0)))

    # Columns
    def column_fn(i):
        column_radius = 0.15
        column_height = base_height * 0.9

        # Position columns at corners
        x_pos = (i % 2) * 2 - 1
        z_pos = (i // 2) * 2 - 1
        x_pos *= (base_width/2 - column_radius)
        z_pos *= (base_depth/2 - column_radius)

        column = primitive_call('cylinder',
                               shape_kwargs={'radius': column_radius,
                                            'p0': (x_pos, -base_height/2 + 0.05, z_pos),
                                            'p1': (x_pos, -base_height/2 + column_height, z_pos)},
                               color=(0.8, 0.8, 0.8))
        return column

    columns = loop(4, column_fn)

    # Steps
    steps_width = base_width * 1.5
    steps_depth = base_depth * 0.4
    steps = primitive_call('cube', shape_kwargs={'scale': (steps_width, 0.2, steps_depth)}, color=(0.75, 0.75, 0.75))
    steps = transform_shape(steps, translation_matrix((0, -base_height/2 - 0.1, base_depth/2 + steps_depth/2)))

    # Decorative elements
    flag_pole = primitive_call('cylinder',
                              shape_kwargs={'radius': 0.05,
                                           'p0': (0, base_height/2 + middle_height + dome_radius*1.2, 0),
                                           'p1': (0, base_height/2 + middle_height + dome_radius*1.2 + 1.0, 0)},
                              color=(0.7, 0.7, 0.7))

    flag = primitive_call('cube', shape_kwargs={'scale': (0.6, 0.4, 0.05)}, color=(0.9, 0.1, 0.1))
    flag = transform_shape(flag, translation_matrix((0.3, base_height/2 + middle_height + dome_radius*1.2 + 0.8, 0)))

    return concat_shapes(base, middle, dome, columns, steps, flag_pole, flag)

@register("Creates a complete city with downtown and suburbs")
def large_scale_city() -> Shape:
    # Create ground with texture variation
    ground_base = primitive_call('cube', shape_kwargs={'scale': (50, 0.1, 50)}, color=(0.3, 0.3, 0.3))
    ground_base = transform_shape(ground_base, translation_matrix((0, -0.05, 0)))  # Position at y=-0.05 so top is at y=0

    # Add texture variation to ground
    def ground_patch_fn(i):
        patch_size = np.random.uniform(3.0, 6.0)
        x_pos = np.random.uniform(-23, 23)
        z_pos = np.random.uniform(-23, 23)

        # Skip patches in city center
        if abs(x_pos) < 10 and abs(z_pos) < 10:
            return []

        # Random earth tone
        color = (np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.3), np.random.uniform(0.1, 0.2))

        patch = primitive_call('cube', shape_kwargs={'scale': (patch_size, 0.11, patch_size)}, color=color)
        return transform_shape(patch, translation_matrix((x_pos, -0.05, z_pos)))

    ground_patches = loop(15, ground_patch_fn)
    ground = concat_shapes(ground_base, ground_patches)

    # Create downtown area with tall buildings
    downtown_size = 3
    downtown_block_size = 5.0
    downtown_road_width = 1.2
    downtown = library_call('city_grid', size=downtown_size, block_size=downtown_block_size, road_width=downtown_road_width, is_downtown=True)

    # Calculate the total size of downtown for proper positioning of suburbs
    downtown_total_size = downtown_size * downtown_block_size + (downtown_size - 1) * downtown_road_width

    # Create suburban areas with proper positioning
    suburb_size = 2
    suburb_block_size = 6.0
    suburb_road_width = 1.0
    suburb_total_size = suburb_size * suburb_block_size + (suburb_size - 1) * suburb_road_width

    # Calculate positions for suburbs to connect with downtown
    suburb_offset = (downtown_total_size + suburb_total_size) / 2

    suburb1 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb1 = transform_shape(suburb1, translation_matrix((suburb_offset, 0, suburb_offset)))

    suburb2 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb2 = transform_shape(suburb2, translation_matrix((-suburb_offset, 0, suburb_offset)))

    suburb3 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb3 = transform_shape(suburb3, translation_matrix((suburb_offset, 0, -suburb_offset)))

    suburb4 = library_call('city_grid', size=suburb_size, block_size=suburb_block_size, road_width=suburb_road_width)
    suburb4 = transform_shape(suburb4, translation_matrix((-suburb_offset, 0, -suburb_offset)))

    # Create connecting roads with proper lengths
    # Calculate the gap between downtown and suburbs
    gap_size = suburb_offset - downtown_total_size/2

    # Horizontal connecting roads
    road1 = library_call('road', length=gap_size, width=1.5)
    road1 = transform_shape(road1, translation_matrix((downtown_total_size/2 + gap_size/2, 0, 0)))

    road2 = library_call('road', length=gap_size, width=1.5)
    road2 = transform_shape(road2, translation_matrix((-downtown_total_size/2 - gap_size/2, 0, 0)))

    # Vertical connecting roads
    road3 = library_call('road', length=gap_size, width=1.5)
    road3 = transform_shape(road3, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road3 = transform_shape(road3, translation_matrix((0, 0, downtown_total_size/2 + gap_size/2)))

    road4 = library_call('road', length=gap_size, width=1.5)
    road4 = transform_shape(road4, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road4 = transform_shape(road4, translation_matrix((0, 0, -downtown_total_size/2 - gap_size/2)))

    # Add parking lots near suburbs
    parking1 = library_call('parking_lot', width=4.0, depth=6.0)
    parking1 = transform_shape(parking1, translation_matrix((suburb_offset - suburb_total_size/2 - 3, 0, suburb_offset)))

    parking2 = library_call('parking_lot', width=4.0, depth=6.0)
    parking2 = transform_shape(parking2, translation_matrix((-suburb_offset + suburb_total_size/2 + 3, 0, -suburb_offset)))

    # Add water features
    water1 = library_call('water_feature', width=8.0, depth=5.0)
    water1 = transform_shape(water1, translation_matrix((15, 0, -15)))

    water2 = library_call('water_feature', width=4.0, depth=10.0)
    water2 = transform_shape(water2, translation_matrix((-18, 0, 5)))

    # Add landmark building in downtown
    landmark = library_call('landmark_building')
    landmark = transform_shape(landmark, translation_matrix((0, 0, 0)))

    return concat_shapes(ground, downtown, suburb1, suburb2, suburb3, suburb4,
                         road1, road2, road3, road4, parking1, parking2,
                         water1, water2, landmark)
```

Code Proposal 3:
```python
from helper import *

"""
a large-scale city
"""

@register("Creates a building with specified dimensions and color")
def building(width: float, height: float, depth: float, color: tuple = (0.7, 0.7, 0.7)) -> Shape:
    building_shape = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    # Position building so its base is at y=0
    return transform_shape(building_shape, translation_matrix((0, height/2, 0)))

@register("Creates a skyscraper with windows")
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    main_building = transform_shape(main_building, translation_matrix((0, height/2, 0)))

    # Add windows - scale number of windows with building height
    windows = []
    window_width = width * 0.15
    window_height = height * 0.05
    window_depth = 0.01
    window_color = (0.9, 0.9, 0.7)

    # Calculate number of floors based on height
    num_floors = max(3, int(height * 2))
    windows_per_floor = 4

    def window_loop_fn(i):
        if i >= num_floors * windows_per_floor:
            return []

        row = i // windows_per_floor
        col = i % windows_per_floor
        window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
        x_pos = (col - 1.5) * (width * 0.22)
        y_pos = (row * height / num_floors) + (height * 0.1)
        z_pos = depth / 2 + window_depth/2  # Position windows on the surface
        return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

    windows = loop(num_floors * windows_per_floor, window_loop_fn)

    return concat_shapes(main_building, windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.8, 0.6, 0.5))
    main_house = transform_shape(main_house, translation_matrix((0, height/2, 0)))

    # Roof
    roof_height = height * 0.5
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=(0.6, 0.3, 0.2))
    roof = transform_shape(roof, translation_matrix((0, height + roof_height/2, 0)))

    # Door
    door_width = width * 0.2
    door_height = height * 0.4
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, 0.01)}, color=(0.4, 0.2, 0.1))
    door = transform_shape(door, translation_matrix((0, door_height/2, depth/2 + 0.005)))  # Position door on the surface

    # Windows
    window1 = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, height * 0.15, 0.01)}, color=(0.8, 0.8, 1.0))
    window1 = transform_shape(window1, translation_matrix((width * 0.25, height * 0.5, depth/2 + 0.005)))  # Position window on the surface

    window2 = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, height * 0.15, 0.01)}, color=(0.8, 0.8, 1.0))
    window2 = transform_shape(window2, translation_matrix((-width * 0.25, height * 0.5, depth/2 + 0.005)))  # Position window on the surface

    return concat_shapes(main_house, roof, door, window1, window2)

@register("Creates a road segment with proper markings")
def road_segment(length: float, width: float = 1.0) -> Shape:
    road = primitive_call('cube', shape_kwargs={'scale': (width, 0.01, length)}, color=(0.2, 0.2, 0.2))
    road = transform_shape(road, translation_matrix((0, 0.005, 0)))  # Position road slightly above ground

    # Road markings - scale with road length
    marking_width = width * 0.05
    marking_length = length * 0.1
    num_markings = max(3, int(length / 0.5))  # Scale number of markings with road length

    def marking_loop_fn(i):
        if i >= num_markings:
            return []

        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, 0.02, marking_length)}, color=(1.0, 1.0, 1.0))
        z_pos = (i - num_markings/2) * (length / num_markings)
        return transform_shape(marking, translation_matrix((0, 0.01, z_pos)))  # Position markings slightly above road

    markings = loop(num_markings, marking_loop_fn)

    # Add sidewalks on both sides of the road
    sidewalk_width = width * 0.3
    sidewalk_left = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, 0.02, length)}, color=(0.8, 0.8, 0.8))
    sidewalk_left = transform_shape(sidewalk_left, translation_matrix((-width/2 - sidewalk_width/2, 0.01, 0)))

    sidewalk_right = primitive_call('cube', shape_kwargs={'scale': (sidewalk_width, 0.02, length)}, color=(0.8, 0.8, 0.8))
    sidewalk_right = transform_shape(sidewalk_right, translation_matrix((width/2 + sidewalk_width/2, 0.01, 0)))

    return concat_shapes(road, markings, sidewalk_left, sidewalk_right)

@register("Creates a traffic light")
def traffic_light() -> Shape:
    # Pole - increased height for better scale
    pole = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 2.0, 0)}, color=(0.3, 0.3, 0.3))

    # Light housing
    housing_height = 0.5
    housing = primitive_call('cube', shape_kwargs={'scale': (0.2, housing_height, 0.2)}, color=(0.2, 0.2, 0.2))
    housing_y = 2.0 + housing_height/2
    housing = transform_shape(housing, translation_matrix((0, housing_y, 0)))

    # Lights - positioned relative to housing
    light_spacing = housing_height / 3

    red_light = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(1.0, 0.0, 0.0))
    red_light = transform_shape(red_light, translation_matrix((0, housing_y + housing_height/3, 0.11)))

    yellow_light = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(1.0, 1.0, 0.0))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, housing_y, 0.11)))

    green_light = primitive_call('sphere', shape_kwargs={'radius': 0.06}, color=(0.0, 1.0, 0.0))
    green_light = transform_shape(green_light, translation_matrix((0, housing_y - housing_height/3, 0.11)))

    return concat_shapes(pole, housing, red_light, yellow_light, green_light)

@register("Creates a street lamp")
def street_lamp() -> Shape:
    # Pole - increased height for better scale
    pole = primitive_call('cylinder', shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, 2.5, 0)}, color=(0.3, 0.3, 0.3))

    # Lamp head
    lamp_head = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 0.9, 0.6))
    lamp_head = transform_shape(lamp_head, translation_matrix((0, 2.7, 0)))

    # Lamp arm
    lamp_arm = primitive_call('cylinder', shape_kwargs={'radius': 0.03, 'p0': (0, 2.3, 0), 'p1': (0, 2.7, 0)}, color=(0.3, 0.3, 0.3))

    return concat_shapes(pole, lamp_head, lamp_arm)

@register("Creates a commercial building with distinctive features")
def commercial_building(width: float, height: float, depth: float) -> Shape:
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.4, 0.6, 0.8))
    main_building = transform_shape(main_building, translation_matrix((0, height/2, 0)))

    # Glass facade
    facade = primitive_call('cube', shape_kwargs={'scale': (width * 0.95, height * 0.9, 0.05)}, color=(0.6, 0.8, 0.9))
    facade = transform_shape(facade, translation_matrix((0, height/2, depth/2 + 0.025)))

    # Entrance
    entrance_width = width * 0.3
    entrance_height = height * 0.2
    entrance = primitive_call('cube', shape_kwargs={'scale': (entrance_width, entrance_height, 0.1)}, color=(0.2, 0.2, 0.3))
    entrance = transform_shape(entrance, translation_matrix((0, entrance_height/2, depth/2 + 0.05)))

    return concat_shapes(main_building, facade, entrance)

@register("Creates a parking lot")
def parking_lot(width: float, depth: float) -> Shape:
    # Base asphalt
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.01, depth)}, color=(0.3, 0.3, 0.3))
    base = transform_shape(base, translation_matrix((0, 0.005, 0)))

    # Parking lines
    spaces_x = max(2, int(width / 0.8))
    spaces_z = max(1, int(depth / 2.0))

    parking_lines = []
    space_width = width / spaces_x
    space_depth = depth / spaces_z

    for i in range(spaces_x):
        for j in range(spaces_z):
            line = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.01, space_depth * 0.8)}, color=(1.0, 1.0, 1.0))
            x_pos = (i - spaces_x/2 + 0.5) * space_width
            z_pos = (j - spaces_z/2 + 0.5) * space_depth
            line = transform_shape(line, translation_matrix((x_pos, 0.01, z_pos)))
            parking_lines.append(line)

    return concat_shapes(base, *parking_lines)

@register("Creates a city block with buildings in a grid pattern")
def city_block(block_size: float, num_buildings: int, block_type: int = 1) -> Shape:
    # Calculate grid dimensions
    grid_size = int(math.sqrt(num_buildings))
    if grid_size * grid_size < num_buildings:
        grid_size += 1

    # Track occupied positions to prevent overlaps
    occupied_positions = []
    buildings = []

    def building_loop_fn(i):
        if i >= num_buildings:
            return []

        # Determine building type based on block type and position
        if block_type == 0:  # Downtown/commercial
            building_type = 0 if i % 5 < 3 else 3  # More skyscrapers and commercial buildings
        elif block_type == 1:  # Mixed use
            building_type = i % 4
        else:  # Residential
            building_type = 2 if i % 3 < 2 else 1  # More houses

        row = i // grid_size
        col = i % grid_size

        # Calculate position in grid
        spacing = block_size / (grid_size + 1)
        x_pos = (col - (grid_size-1)/2) * spacing
        z_pos = (row - (grid_size-1)/2) * spacing

        # Add slight randomness to position
        x_offset = math.sin(i * 0.5) * spacing * 0.1
        z_offset = math.cos(i * 0.5) * spacing * 0.1
        x_pos += x_offset
        z_pos += z_offset

        # Determine building size based on type
        if building_type == 0:  # Skyscraper
            width = 0.3 + (i % 5) * 0.05
            height = 4.0 + (i % 7) * 0.5  # Taller skyscrapers
            depth = 0.3 + (i % 4) * 0.05
            building_shape = library_call('skyscraper', width=width, height=height, depth=depth)
        elif building_type == 1:  # Regular building
            width = 0.4 + (i % 3) * 0.05
            height = 1.5 + (i % 4) * 0.2
            depth = 0.4 + (i % 3) * 0.05
            building_shape = library_call('building', width=width, height=height, depth=depth)
        elif building_type == 2:  # House
            width = 0.3 + (i % 2) * 0.05
            height = 0.5 + (i % 3) * 0.1
            depth = 0.3 + (i % 2) * 0.05
            building_shape = library_call('house', width=width, height=height, depth=depth)
        else:  # Commercial building
            width = 0.5 + (i % 3) * 0.1
            height = 2.0 + (i % 3) * 0.3
            depth = 0.5 + (i % 3) * 0.1
            building_shape = library_call('commercial_building', width=width, height=height, depth=depth)

        # Check for collisions with existing buildings
        building_footprint = (width, depth)
        for pos, size in occupied_positions:
            if (abs(pos[0] - x_pos) < (building_footprint[0]/2 + size[0]/2 + 0.05) and
                abs(pos[1] - z_pos) < (building_footprint[1]/2 + size[1]/2 + 0.05)):
                # Skip this building due to overlap
                return []

        # Add to occupied positions
        occupied_positions.append(((x_pos, z_pos), building_footprint))

        return transform_shape(building_shape, translation_matrix((x_pos, 0, z_pos)))

    buildings = loop(grid_size * grid_size, building_loop_fn)

    # Add parking lot if this is a commercial block
    if block_type == 0 or block_type == 1:
        parking = library_call('parking_lot', width=block_size * 0.3, depth=block_size * 0.2)
        parking = transform_shape(parking, translation_matrix((block_size * 0.3, 0, block_size * 0.3)))
        buildings = concat_shapes(buildings, parking)

    # Add street furniture
    street_furniture = []

    # Add traffic lights at corners
    traffic_light1 = library_call('traffic_light')
    traffic_light1 = transform_shape(traffic_light1, translation_matrix((block_size/2, 0, block_size/2)))

    traffic_light2 = library_call('traffic_light')
    traffic_light2 = transform_shape(traffic_light2, translation_matrix((-block_size/2, 0, block_size/2)))

    traffic_light3 = library_call('traffic_light')
    traffic_light3 = transform_shape(traffic_light3, translation_matrix((block_size/2, 0, -block_size/2)))

    traffic_light4 = library_call('traffic_light')
    traffic_light4 = transform_shape(traffic_light4, translation_matrix((-block_size/2, 0, -block_size/2)))

    # Add street lamps along block perimeter
    def lamp_loop_fn(i):
        if i >= 8:
            return []

        # Place lamps along the perimeter of the block
        if i < 2:  # North side
            x = (i - 0.5) * block_size
            z = -block_size/2 + 0.2
        elif i < 4:  # East side
            x = block_size/2 - 0.2
            z = (i - 3) * block_size
        elif i < 6:  # South side
            x = (i - 5.5) * block_size
            z = block_size/2 - 0.2
        else:  # West side
            x = -block_size/2 + 0.2
            z = (i - 8) * block_size

        lamp = library_call('street_lamp')
        return transform_shape(lamp, translation_matrix((x, 0, z)))

    street_lamps = loop(8, lamp_loop_fn)

    street_furniture = concat_shapes(traffic_light1, traffic_light2, traffic_light3, traffic_light4, street_lamps)

    return concat_shapes(buildings, street_furniture)

@register("Creates a park with trees and features")
def park(size: float) -> Shape:
    # Base grass
    grass = primitive_call('cube', shape_kwargs={'scale': (size, 0.01, size)}, color=(0.2, 0.7, 0.2))
    grass = transform_shape(grass, translation_matrix((0, 0.005, 0)))  # Position grass slightly above ground

    # Trees with more variety and natural placement
    def tree_loop_fn(i):
        if i >= 15:  # More trees for a denser park
            return []

        # Use polar coordinates for more natural tree placement
        angle = i * math.pi * 0.7  # Non-uniform angles
        distance = (0.2 + (i % 5) * 0.15) * size/2  # Varied distances from center

        x_pos = math.cos(angle) * distance
        z_pos = math.sin(angle) * distance

        # Tree trunk
        trunk_height = 0.5 + (i % 5) * 0.1  # Taller trees
        trunk = primitive_call('cylinder', shape_kwargs={
            'radius': 0.05 + (i % 3) * 0.01,  # Varied trunk thickness
            'p0': (0, 0, 0),
            'p1': (0, trunk_height, 0)
        }, color=(0.5, 0.3, 0.1))

        # Tree top - vary the shape based on tree type
        if i % 3 == 0:  # Conical tree
            top = primitive_call('cube', shape_kwargs={'scale': (0.3, 0.4, 0.3)}, color=(0.1, 0.5, 0.1))
            top = transform_shape(top, translation_matrix((0, trunk_height + 0.2, 0)))
        else:  # Round tree
            top_radius = 0.15 + (i % 3) * 0.05
            top = primitive_call('sphere', shape_kwargs={'radius': top_radius}, color=(0.1 + (i % 3) * 0.05, 0.5, 0.1))
            top = transform_shape(top, translation_matrix((0, trunk_height + top_radius * 0.5, 0)))

        tree = concat_shapes(trunk, top)
        return transform_shape(tree, translation_matrix((x_pos, 0, z_pos)))

    trees = loop(15, tree_loop_fn)

    # Add benches (more proportional)
    def bench_loop_fn(i):
        if i >= 4:
            return []

        angle = i * math.pi/2 + math.pi/4  # Place benches at 45°, 135°, 225°, 315°
        distance = size * 0.35

        x_pos = math.cos(angle) * distance
        z_pos = math.sin(angle) * distance

        # Bench seat
        seat = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.05, 0.15)}, color=(0.6, 0.4, 0.2))

        # Bench legs
        leg1 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, 0.15)}, color=(0.5, 0.3, 0.1))
        leg1 = transform_shape(leg1, translation_matrix((0.15, -0.05, 0)))

        leg2 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, 0.15)}, color=(0.5, 0.3, 0.1))
        leg2 = transform_shape(leg2, translation_matrix((-0.15, -0.05, 0)))

        bench = concat_shapes(seat, leg1, leg2)
        bench = transform_shape(bench, translation_matrix((x_pos, 0.1, z_pos)))
        bench = transform_shape(bench, rotation_matrix(angle + math.pi/2, (0, 1, 0), (x_pos, 0.1, z_pos)))

        return bench

    benches = loop(4, bench_loop_fn)

    # Add a larger fountain
    fountain_base = primitive_call('cylinder', shape_kwargs={'radius': 0.4, 'p0': (0, 0, 0), 'p1': (0, 0.1, 0)}, color=(0.7, 0.7, 0.7))
    fountain_rim = primitive_call('cylinder', shape_kwargs={'radius': 0.45, 'p0': (0, 0.1, 0), 'p1': (0, 0.15, 0)}, color=(0.6, 0.6, 0.6))
    fountain_water = primitive_call('cylinder', shape_kwargs={'radius': 0.35, 'p0': (0, 0.1, 0), 'p1': (0, 0.12, 0)}, color=(0.6, 0.8, 1.0))

    # Fountain center piece
    center_column = primitive_call('cylinder', shape_kwargs={'radius': 0.08, 'p0': (0, 0, 0), 'p1': (0, 0.3, 0)}, color=(0.6, 0.6, 0.6))
    water_spray = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.7, 0.9, 1.0))
    water_spray = transform_shape(water_spray, translation_matrix((0, 0.35, 0)))

    fountain = concat_shapes(fountain_base, fountain_rim, fountain_water, center_column, water_spray)
    fountain = transform_shape(fountain, translation_matrix((0, 0, 0)))

    # Add pathways
    def pathway_loop_fn(i):
        if i >= 4:
            return []

        angle = i * math.pi/2
        path = primitive_call('cube', shape_kwargs={'scale': (0.2, 0.02, size/2)}, color=(0.8, 0.7, 0.6))
        path = transform_shape(path, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        path = transform_shape(path, translation_matrix((0, 0.01, 0)))
        return path

    pathways = loop(4, pathway_loop_fn)

    return concat_shapes(grass, trees, benches, fountain, pathways)

@register("Creates a bridge over a road")
def bridge(length: float, width: float) -> Shape:
    # Bridge deck
    deck = primitive_call('cube', shape_kwargs={'scale': (width, 0.2, length)}, color=(0.5, 0.5, 0.5))
    deck = transform_shape(deck, translation_matrix((0, 1.5, 0)))  # Elevated above road

    # Support columns
    column1 = primitive_call('cylinder', shape_kwargs={'radius': 0.2, 'p0': (0, 0, -length/3), 'p1': (0, 1.5, -length/3)}, color=(0.4, 0.4, 0.4))
    column2 = primitive_call('cylinder', shape_kwargs={'radius': 0.2, 'p0': (0, 0, length/3), 'p1': (0, 1.5, length/3)}, color=(0.4, 0.4, 0.4))

    # Railings
    railing1 = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, 0.05)}, color=(0.3, 0.3, 0.3))
    railing1 = transform_shape(railing1, translation_matrix((0, 1.65, -length/2 + 0.025)))

    railing2 = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, 0.05)}, color=(0.3, 0.3, 0.3))
    railing2 = transform_shape(railing2, translation_matrix((0, 1.65, length/2 - 0.025)))

    railing3 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, length)}, color=(0.3, 0.3, 0.3))
    railing3 = transform_shape(railing3, translation_matrix((width/2 - 0.025, 1.65, 0)))

    railing4 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.1, length)}, color=(0.3, 0.3, 0.3))
    railing4 = transform_shape(railing4, translation_matrix((-width/2 + 0.025, 1.65, 0)))

    return concat_shapes(deck, column1, column2, railing1, railing2, railing3, railing4)

@register("Creates a connected road network for the city with intersections")
def road_network(size: int, block_size: float, spacing: float) -> Shape:
    total_size = size * (block_size + spacing)
    roads = []

    # Create horizontal roads
    for i in range(size + 1):
        pos = (i - size/2) * (block_size + spacing)
        road = library_call('road_segment', length=total_size, width=spacing)
        road = transform_shape(road, translation_matrix((0, 0, pos)))
        roads.append(road)

    # Create vertical roads
    for i in range(size + 1):
        pos = (i - size/2) * (block_size + spacing)
        road = library_call('road_segment', length=total_size, width=spacing)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((pos, 0, 0)))
        roads.append(road)

    # Create intersections
    for i in range(size + 1):
        for j in range(size + 1):
            x_pos = (i - size/2) * (block_size + spacing)
            z_pos = (j - size/2) * (block_size + spacing)

            intersection = primitive_call('cube', shape_kwargs={'scale': (spacing, 0.015, spacing)}, color=(0.25, 0.25, 0.25))
            intersection = transform_shape(intersection, translation_matrix((x_pos, 0.007, z_pos)))
            roads.append(intersection)

    # Add a bridge over one of the roads
    if size >= 3:
        bridge_shape = library_call('bridge', length=block_size + spacing, width=spacing)
        bridge_shape = transform_shape(bridge_shape, translation_matrix((0, 0, (block_size + spacing))))
        roads.append(bridge_shape)

    return concat_shapes(*roads)

@register("Creates a large-scale city with multiple blocks and connected roads")
def city(size: int = 4) -> Shape:
    block_size = 3.0  # Increased block size for better proportions
    road_width = 1.0  # Wider roads
    block_spacing = road_width  # Space between blocks equals road width

    # Create ground
    ground_size = size * (block_size + block_spacing) + block_spacing
    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, 0.1, ground_size)}, color=(0.6, 0.6, 0.6))
    ground = transform_shape(ground, translation_matrix((0, -0.05, 0)))  # Position ground so its top is at y=0

    # Create connected road network
    roads = library_call('road_network', size=size, block_size=block_size, spacing=block_spacing)

    # Create city blocks with different types based on location
    def city_block_loop_fn(i):
        if i >= size * size:
            return []

        row = i // size
        col = i % size

        # Determine block type based on position
        # 0: Downtown/commercial, 1: Mixed use, 2: Residential, 3: Park
        distance_from_center = math.sqrt((row - size/2 + 0.5)**2 + (col - size/2 + 0.5)**2)

        if distance_from_center < size/4:
            block_type = 0  # Downtown
            num_buildings = 12
        elif distance_from_center < size/2:
            block_type = 1  # Mixed use
            num_buildings = 9
        elif i % 7 != 0:  # Most outer blocks are residential
            block_type = 2  # Residential
            num_buildings = 6
        else:  # Some outer blocks are parks
            block_type = 3  # Park

        if block_type < 3:
            block = library_call('city_block', block_size=block_size, num_buildings=num_buildings, block_type=block_type)
        else:
            block = library_call('park', size=block_size)

        x_pos = (col - size/2 + 0.5) * (block_size + block_spacing)
        z_pos = (row - size/2 + 0.5) * (block_size + block_spacing)

        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))

    city_blocks = loop(size * size, city_block_loop_fn)

    return concat_shapes(ground, roads, city_blocks)
```

Code Proposal 4:
```python
from helper import *

@register()
def building(width: float, height: float, depth: float, color: tuple = (0.7, 0.7, 0.7)) -> Shape:
    """Creates a building with specified dimensions and color"""
    return primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)

@register()
def skyscraper(width: float, height: float, depth: float, color: tuple = (0.5, 0.5, 0.6)) -> Shape:
    """Creates a skyscraper with windows"""
    # Main building
    main_building = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=color)
    all_shapes = [main_building]

    # Add windows
    window_width = width * 0.1
    window_height = width * 0.15
    window_depth = 0.01
    window_spacing_h = width / 8
    window_spacing_v = height / 15

    # Calculate how many windows can fit
    num_windows_h = max(1, int(width / window_spacing_h) - 1)
    num_windows_v = max(1, int(height / window_spacing_v) - 1)

    # Create windows on front face
    for i in range(num_windows_h):
        for j in range(num_windows_v):
            x_pos = (i - (num_windows_h - 1) / 2) * window_spacing_h
            y_pos = (j - (num_windows_v - 1) / 2) * window_spacing_v
            window = primitive_call('cube',
                                   shape_kwargs={'scale': (window_width, window_height, window_depth)},
                                   color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((x_pos, y_pos, depth/2 + window_depth/2)))
            all_shapes.append(window)

    # Create windows on back face
    for i in range(num_windows_h):
        for j in range(num_windows_v):
            x_pos = (i - (num_windows_h - 1) / 2) * window_spacing_h
            y_pos = (j - (num_windows_v - 1) / 2) * window_spacing_v
            window = primitive_call('cube',
                                   shape_kwargs={'scale': (window_width, window_height, window_depth)},
                                   color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((x_pos, y_pos, -depth/2 - window_depth/2)))
            all_shapes.append(window)

    # Add an antenna on top
    antenna_height = height * 0.2
    antenna = primitive_call('cylinder',
                           shape_kwargs={'radius': 0.05, 'p0': (0, height/2, 0), 'p1': (0, height/2 + antenna_height, 0)},
                           color=(0.3, 0.3, 0.3))
    all_shapes.append(antenna)

    return concat_shapes(*all_shapes)

@register()
def house(width: float, height: float, depth: float) -> Shape:
    """Creates a house with a roof"""
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.8, 0.7, 0.6))

    # Roof
    roof_height = height * 0.5
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=(0.6, 0.3, 0.2))

    # Position roof on top of house
    house_max = compute_shape_max(main_house)
    roof = transform_shape(roof, translation_matrix((0, house_max[1] + roof_height/2, 0)))

    # Add a door
    door_width = width * 0.2
    door_height = height * 0.4
    door_depth = 0.01
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, door_depth)}, color=(0.4, 0.2, 0.1))

    # Position door at the bottom front of the house
    house_min = compute_shape_min(main_house)
    door = transform_shape(door, translation_matrix((0, house_min[1] + door_height/2, depth/2 + door_depth/2)))

    return concat_shapes(main_house, roof, door)

@register()
def road(length: float, width: float) -> Shape:
    """Creates a road segment"""
    road_base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, length)}, color=(0.2, 0.2, 0.2))
    all_shapes = [road_base]

    # Add road markings
    marking_width = width * 0.05
    marking_length = length * 0.1
    marking_height = 0.01

    num_markings = int(length / (marking_length * 2))

    for i in range(num_markings):
        z_pos = (i - (num_markings - 1) / 2) * marking_length * 2
        marking = primitive_call('cube',
                               shape_kwargs={'scale': (marking_width, marking_height, marking_length)},
                               color=(1.0, 1.0, 1.0))
        marking = transform_shape(marking, translation_matrix((0, 0.05 + marking_height/2, z_pos)))
        all_shapes.append(marking)

    return concat_shapes(*all_shapes)

@register()
def street_lamp(height: float) -> Shape:
    """Creates a street lamp"""
    # Pole
    pole = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, height, 0)},
                         color=(0.3, 0.3, 0.3))

    # Light bulb
    light = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 0.8))
    light = transform_shape(light, translation_matrix((0, height + 0.15, 0)))

    # Light fixture - horizontal cylinder
    fixture = primitive_call('cylinder',
                           shape_kwargs={'radius': 0.1, 'p0': (-0.2, height, 0), 'p1': (0.2, height, 0)},
                           color=(0.2, 0.2, 0.2))

    return concat_shapes(pole, light, fixture)

@register()
def tree(height: float) -> Shape:
    """Creates a tree"""
    # Trunk
    trunk_height = height * 0.3
    trunk = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.1, 'p0': (0, 0, 0), 'p1': (0, trunk_height, 0)},
                          color=(0.5, 0.3, 0.2))

    # Foliage
    foliage_radius = height * 0.2
    foliage = primitive_call('sphere', shape_kwargs={'radius': foliage_radius}, color=(0.1, 0.6, 0.1))
    foliage = transform_shape(foliage, translation_matrix((0, trunk_height + foliage_radius * 0.7, 0)))

    return concat_shapes(trunk, foliage)

@register()
def park_bench(width: float = 0.8) -> Shape:
    """Creates a park bench"""
    # Bench seat
    seat = primitive_call('cube', shape_kwargs={'scale': (width, 0.05, 0.3)}, color=(0.4, 0.2, 0.1))
    seat = transform_shape(seat, translation_matrix((0, 0.25, 0)))

    # Bench legs
    leg1 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.25, 0.3)}, color=(0.3, 0.3, 0.3))
    leg1 = transform_shape(leg1, translation_matrix((width/2 - 0.05, 0.125, 0)))

    leg2 = primitive_call('cube', shape_kwargs={'scale': (0.05, 0.25, 0.3)}, color=(0.3, 0.3, 0.3))
    leg2 = transform_shape(leg2, translation_matrix((-width/2 + 0.05, 0.125, 0)))

    # Bench back
    back = primitive_call('cube', shape_kwargs={'scale': (width, 0.3, 0.05)}, color=(0.4, 0.2, 0.1))
    back = transform_shape(back, translation_matrix((0, 0.4, -0.125)))

    return concat_shapes(seat, leg1, leg2, back)

@register()
def city_block(width: float, depth: float) -> Shape:
    """Creates a city block with buildings"""
    buildings_list = []

    # Number of buildings in each direction
    num_buildings_x = 3
    num_buildings_z = 3

    # Space between buildings
    spacing_x = width / num_buildings_x
    spacing_z = depth / num_buildings_z

    for i in range(num_buildings_x):
        for j in range(num_buildings_z):
            # Randomize building properties
            building_width = spacing_x * 0.7
            building_depth = spacing_z * 0.7
            building_height = np.random.uniform(1.0, 3.0)

            # Determine building type based on random value
            rand_val = np.random.random()
            if rand_val < 0.4:
                building_type = 'building'
            elif rand_val < 0.8:
                building_type = 'skyscraper'
            else:
                building_type = 'house'

            # Calculate position
            x_pos = (i - (num_buildings_x - 1) / 2) * spacing_x
            z_pos = (j - (num_buildings_z - 1) / 2) * spacing_z

            # Create building based on type
            if building_type == 'building':
                r = 0.5 + 0.3 * np.random.random()
                g = 0.5 + 0.3 * np.random.random()
                b = 0.5 + 0.3 * np.random.random()
                building = library_call('building', width=building_width, height=building_height, depth=building_depth, color=(r, g, b))
            elif building_type == 'skyscraper':
                building_height *= 2  # Skyscrapers are taller
                r = 0.4 + 0.2 * np.random.random()
                g = 0.4 + 0.2 * np.random.random()
                b = 0.5 + 0.2 * np.random.random()
                building = library_call('skyscraper', width=building_width, height=building_height, depth=building_depth, color=(r, g, b))
            else:  # house
                building_height *= 0.5  # Houses are shorter
                building = library_call('house', width=building_width, height=building_height, depth=building_depth)

            # Position the building
            building = transform_shape(building, translation_matrix((x_pos, building_height/2, z_pos)))
            buildings_list.append(building)

    return concat_shapes(*buildings_list)

@register()
def park(width: float, depth: float) -> Shape:
    """Creates a park with trees and benches"""
    # Base ground
    ground = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))
    all_shapes = [ground]

    # Add a pond in the center
    pond_radius = min(width, depth) * 0.2
    pond = primitive_call('cylinder',
                         shape_kwargs={'radius': pond_radius, 'p0': (0, 0.05, 0), 'p1': (0, 0.06, 0)},
                         color=(0.1, 0.4, 0.8))
    all_shapes.append(pond)

    # Add trees
    num_trees = 10
    placed_trees = 0
    attempts = 0
    while placed_trees < num_trees and attempts < 100:  # Prevent infinite loop
        attempts += 1
        x_pos = np.random.random() * (width - 1.0) - (width/2 - 0.5)
        z_pos = np.random.random() * (depth - 1.0) - (depth/2 - 0.5)

        # Avoid placing trees in the pond
        if x_pos*x_pos + z_pos*z_pos < pond_radius*pond_radius:
            continue

        tree_height = 0.8 + 0.7 * np.random.random()
        tree = library_call('tree', height=tree_height)
        tree = transform_shape(tree, translation_matrix((x_pos, 0.05, z_pos)))
        all_shapes.append(tree)
        placed_trees += 1

    # Add benches
    num_benches = 4
    for i in range(num_benches):
        angle = i * (2 * math.pi / num_benches)
        x_pos = (pond_radius + 0.5) * math.cos(angle)
        z_pos = (pond_radius + 0.5) * math.sin(angle)

        bench = library_call('park_bench')
        # Rotate bench to face the pond
        bench = transform_shape(bench, rotation_matrix(angle + math.pi, (0, 1, 0), (0, 0, 0)))
        bench = transform_shape(bench, translation_matrix((x_pos, 0.05, z_pos)))
        all_shapes.append(bench)

    # Add paths
    path_width = 0.5
    for i in range(4):
        angle = i * (math.pi / 2)
        path_length = max(width, depth) / 2

        path = primitive_call('cube',
                             shape_kwargs={'scale': (path_width, 0.02, path_length)},
                             color=(0.8, 0.7, 0.6))
        path = transform_shape(path, rotation_matrix(angle, (0, 1, 0), (0, 0, 0)))
        path = transform_shape(path, translation_matrix((0, 0.06, 0)))
        all_shapes.append(path)

    return concat_shapes(*all_shapes)

@register()
def landmark_building() -> Shape:
    """Creates a landmark building"""
    # Base
    base_width = 3.0
    base_height = 1.0
    base_depth = 3.0
    base = primitive_call('cube', shape_kwargs={'scale': (base_width, base_height, base_depth)}, color=(0.7, 0.7, 0.8))

    # Tower
    tower_width = 1.5
    tower_height = 6.0
    tower_depth = 1.5
    tower = primitive_call('cube', shape_kwargs={'scale': (tower_width, tower_height, tower_depth)}, color=(0.6, 0.6, 0.7))
    tower = transform_shape(tower, translation_matrix((0, base_height/2 + tower_height/2, 0)))

    # Spire
    spire_height = 2.0
    spire = primitive_call('cylinder',
                          shape_kwargs={'radius': 0.1, 'p0': (0, base_height/2 + tower_height, 0),
                                       'p1': (0, base_height/2 + tower_height + spire_height, 0)},
                          color=(0.8, 0.8, 0.9))

    # Windows
    windows = []
    window_size = 0.2
    for i in range(5):  # 5 floors of windows
        for j in range(-1, 2):  # 3 windows per side
            # Front windows
            window = primitive_call('cube',
                                  shape_kwargs={'scale': (window_size, window_size, 0.05)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((j*0.4, base_height/2 + i*1.0 + 0.5, tower_depth/2)))
            windows.append(window)

            # Back windows
            window = primitive_call('cube',
                                  shape_kwargs={'scale': (window_size, window_size, 0.05)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((j*0.4, base_height/2 + i*1.0 + 0.5, -tower_depth/2)))
            windows.append(window)

            # Side windows
            window = primitive_call('cube',
                                  shape_kwargs={'scale': (0.05, window_size, window_size)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((tower_width/2, base_height/2 + i*1.0 + 0.5, j*0.4)))
            windows.append(window)

            window = primitive_call('cube',
                                  shape_kwargs={'scale': (0.05, window_size, window_size)},
                                  color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((-tower_width/2, base_height/2 + i*1.0 + 0.5, j*0.4)))
            windows.append(window)

    return concat_shapes(base, tower, spire, *windows)

@register()
def city_district(size: float, seed: int = None) -> Shape:
    """Creates a city district with blocks, roads, parks, and landmarks"""
    if seed is not None:
        np.random.seed(seed)

    district_elements = []

    # Create a grid of city blocks and roads
    grid_size = 3
    block_size = size / grid_size
    road_width = block_size * 0.2  # Reduced from 0.3 for better proportions

    # Create city blocks
    for i in range(grid_size):
        for j in range(grid_size):
            # Center block is a landmark or plaza
            if i == grid_size // 2 and j == grid_size // 2:
                landmark = library_call('landmark_building')
                landmark = transform_shape(landmark, translation_matrix((0, 0, 0)))
                district_elements.append(landmark)
                continue

            # Determine if this should be a park (20% chance)
            is_park = np.random.random() < 0.2

            x_pos = (i - (grid_size - 1) / 2) * block_size
            z_pos = (j - (grid_size - 1) / 2) * block_size

            if is_park:
                block = library_call('park', width=block_size * 0.8, depth=block_size * 0.8)
            else:
                block = library_call('city_block', width=block_size * 0.8, depth=block_size * 0.8)

            block = transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
            district_elements.append(block)

    # Create horizontal roads
    for i in range(grid_size + 1):
        z_pos = (i - grid_size / 2) * block_size
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        road = transform_shape(road, translation_matrix((0, 0, z_pos)))
        district_elements.append(road)

    # Create vertical roads
    for i in range(grid_size + 1):
        x_pos = (i - grid_size / 2) * block_size
        road = library_call('road', length=size, width=road_width)
        road = transform_shape(road, translation_matrix((x_pos, 0, 0)))
        district_elements.append(road)

    # Add street lamps along roads
    lamp_height = 0.5  # Reduced from 1.0 for better proportions
    for i in range(grid_size + 1):
        for j in range(5):  # 5 lamps per road
            # Horizontal roads
            z_pos = (i - grid_size / 2) * block_size
            x_pos = (j - 2) * (size / 4)
            lamp = library_call('street_lamp', height=lamp_height)
            lamp = transform_shape(lamp, translation_matrix((x_pos, 0, z_pos + road_width/3)))
            district_elements.append(lamp)

            # Vertical roads
            x_pos = (i - grid_size / 2) * block_size
            z_pos = (j - 2) * (size / 4)
            lamp = library_call('street_lamp', height=lamp_height)
            lamp = transform_shape(lamp, translation_matrix((x_pos + road_width/3, 0, z_pos)))
            district_elements.append(lamp)

    return concat_shapes(*district_elements)

@register()
def highway(length: float, width: float) -> Shape:
    """Creates a highway segment"""
    # Main road
    highway_base = primitive_call('cube', shape_kwargs={'scale': (width, 0.2, length)}, color=(0.3, 0.3, 0.3))

    # Divider
    divider_width = 0.2
    divider = primitive_call('cube', shape_kwargs={'scale': (divider_width, 0.25, length)}, color=(0.7, 0.7, 0.7))

    # Road markings
    markings = []
    marking_width = 0.1
    marking_length = length * 0.05
    marking_spacing = length * 0.1
    num_markings = int(length / (marking_length + marking_spacing))

    for i in range(num_markings):
        z_pos = (i - (num_markings - 1) / 2) * (marking_length + marking_spacing)

        # Left lane marking
        left_marking = primitive_call('cube',
                                    shape_kwargs={'scale': (marking_width, 0.01, marking_length)},
                                    color=(1.0, 1.0, 0.0))
        left_marking = transform_shape(left_marking, translation_matrix((-width/4, 0.11, z_pos)))
        markings.append(left_marking)

        # Right lane marking
        right_marking = primitive_call('cube',
                                     shape_kwargs={'scale': (marking_width, 0.01, marking_length)},
                                     color=(1.0, 1.0, 0.0))
        right_marking = transform_shape(right_marking, translation_matrix((width/4, 0.11, z_pos)))
        markings.append(right_marking)

    # Guardrails
    left_rail = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.3, length)}, color=(0.6, 0.6, 0.6))
    left_rail = transform_shape(left_rail, translation_matrix((-width/2 - 0.1, 0.15, 0)))

    right_rail = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.3, length)}, color=(0.6, 0.6, 0.6))
    right_rail = transform_shape(right_rail, translation_matrix((width/2 + 0.1, 0.15, 0)))

    return concat_shapes(highway_base, divider, left_rail, right_rail, *markings)

@register()
def city(seed: int = 42) -> Shape:
    """Creates a complete city with multiple districts and highways"""
    np.random.seed(seed)
    city_elements = []

    # Create multiple districts with different seeds
    district_size = 10
    district_spacing = 11  # Reduced spacing to make city more continuous

    for i in range(-1, 2):
        for j in range(-1, 2):
            district = library_call('city_district', size=district_size, seed=seed + i*10 + j)
            district = transform_shape(district, translation_matrix((i * district_spacing, 0, j * district_spacing)))
            city_elements.append(district)

    # Add highways connecting districts
    highway_width = 2.0  # Reduced from 3.0 for better proportions

    # East-West highways
    for j in range(-1, 2):
        highway_segment = library_call('highway', length=district_spacing*3, width=highway_width)
        highway_segment = transform_shape(highway_segment, translation_matrix((0, 0, j * district_spacing - highway_width/2)))
        city_elements.append(highway_segment)

    # North-South highways
    for i in range(-1, 2):
        highway_segment = library_call('highway', length=district_spacing*3, width=highway_width)
        highway_segment = transform_shape(highway_segment, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
        highway_segment = transform_shape(highway_segment, translation_matrix((i * district_spacing - highway_width/2, 0, 0)))
        city_elements.append(highway_segment)

    return concat_shapes(*city_elements)

def main() -> Shape:
    """Main function that returns the complete city"""
    return library_call('city')
```



Output the index of the best code proposal and a rationale for your choice.


![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/judge/expert_renderings.png)

