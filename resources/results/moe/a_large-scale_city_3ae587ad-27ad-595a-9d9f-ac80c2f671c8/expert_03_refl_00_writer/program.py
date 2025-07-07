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
            windows.append(window)

    # Create windows on back face
    for i in range(num_windows_h):
        for j in range(num_windows_v):
            x_pos = (i - (num_windows_h - 1) / 2) * window_spacing_h
            y_pos = (j - (num_windows_v - 1) / 2) * window_spacing_v
            window = primitive_call('cube',
                                   shape_kwargs={'scale': (window_width, window_height, window_depth)},
                                   color=(0.9, 0.9, 1.0))
            window = transform_shape(window, translation_matrix((x_pos, y_pos, -depth/2 - window_depth/2)))
            windows.append(window)

    return concat_shapes(main_building, *windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.8, 0.7, 0.6))

    # Roof
    roof_height = height * 0.5
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=(0.6, 0.3, 0.2))

    # Position roof on top of house
    house_center = compute_shape_center(main_house)
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

@register("Creates a road segment")
def road(length: float, width: float) -> Shape:
    road_base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, length)}, color=(0.2, 0.2, 0.2))

    # Add road markings
    marking_width = width * 0.05
    marking_length = length * 0.1
    marking_height = 0.01

    markings = []
    num_markings = int(length / (marking_length * 2))

    for i in range(num_markings):
        z_pos = (i - (num_markings - 1) / 2) * marking_length * 2
        marking = primitive_call('cube',
                               shape_kwargs={'scale': (marking_width, marking_height, marking_length)},
                               color=(1.0, 1.0, 1.0))
        marking = transform_shape(marking, translation_matrix((0, 0.05 + marking_height/2, z_pos)))
        markings.append(marking)

    return concat_shapes(road_base, *markings)

@register("Creates a street lamp")
def street_lamp(height: float) -> Shape:
    # Pole
    pole = primitive_call('cylinder',
                         shape_kwargs={'radius': 0.05, 'p0': (0, 0, 0), 'p1': (0, height, 0)},
                         color=(0.3, 0.3, 0.3))

    # Light bulb
    light = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(1.0, 1.0, 0.8))
    light = transform_shape(light, translation_matrix((0, height + 0.15, 0)))

    # Light fixture
    fixture = primitive_call('cylinder',
                           shape_kwargs={'radius': 0.2, 'p0': (0, height, 0), 'p1': (0, height + 0.1, 0)},
                           color=(0.2, 0.2, 0.2))

    return concat_shapes(pole, light, fixture)

@register("Creates a tree")
def tree(height: float) -> Shape:
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

@register("Creates a city block with buildings")
def city_block(width: float, depth: float) -> Shape:
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
            building_type = np.random.choice(['building', 'skyscraper', 'house'], p=[0.4, 0.4, 0.2])

            # Calculate position
            x_pos = (i - (num_buildings_x - 1) / 2) * spacing_x
            z_pos = (j - (num_buildings_z - 1) / 2) * spacing_z

            # Create building based on type
            if building_type == 'building':
                color = (np.random.uniform(0.5, 0.8), np.random.uniform(0.5, 0.8), np.random.uniform(0.5, 0.8))
                building = library_call('building', width=building_width, height=building_height, depth=building_depth, color=color)
            elif building_type == 'skyscraper':
                building_height *= 2  # Skyscrapers are taller
                color = (np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7))
                building = library_call('skyscraper', width=building_width, height=building_height, depth=building_depth, color=color)
            else:  # house
                building_height *= 0.5  # Houses are shorter
                building = library_call('house', width=building_width, height=building_height, depth=building_depth)

            # Position the building
            building = transform_shape(building, translation_matrix((x_pos, building_height/2, z_pos)))
            buildings_list.append(building)

    return concat_shapes(*buildings_list)

@register("Creates a park with trees")
def park(width: float, depth: float) -> Shape:
    # Base ground
    ground = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))

    # Add trees
    trees_list = []
    num_trees = 10

    for _ in range(num_trees):
        x_pos = np.random.uniform(-width/2 + 0.5, width/2 - 0.5)
        z_pos = np.random.uniform(-depth/2 + 0.5, depth/2 - 0.5)
        tree_height = np.random.uniform(0.8, 1.5)

        tree = library_call('tree', height=tree_height)
        tree = transform_shape(tree, translation_matrix((x_pos, 0.05, z_pos)))
        trees_list.append(tree)

    return concat_shapes(ground, *trees_list)

@register("Creates a city district with blocks, roads, and parks")
def city_district(size: float) -> Shape:
    district_elements = []

    # Create a grid of city blocks and roads
    grid_size = 3
    block_size = size / grid_size
    road_width = block_size * 0.3

    # Create city blocks
    for i in range(grid_size):
        for j in range(grid_size):
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
    for i in range(grid_size + 1):
        for j in range(5):  # 5 lamps per road
            # Horizontal roads
            z_pos = (i - grid_size / 2) * block_size
            x_pos = (j - 2) * (size / 4)
            lamp = library_call('street_lamp', height=1.0)
            lamp = transform_shape(lamp, translation_matrix((x_pos, 0, z_pos + road_width/3)))
            district_elements.append(lamp)

            # Vertical roads
            x_pos = (i - grid_size / 2) * block_size
            z_pos = (j - 2) * (size / 4)
            lamp = library_call('street_lamp', height=1.0)
            lamp = transform_shape(lamp, translation_matrix((x_pos + road_width/3, 0, z_pos)))
            district_elements.append(lamp)

    return concat_shapes(*district_elements)

@register("Creates a complete city with multiple districts")
def city() -> Shape:
    city_elements = []

    # Create multiple districts
    for i in range(-1, 2):
        for j in range(-1, 2):
            district = library_call('city_district', size=10)
            district = transform_shape(district, translation_matrix((i * 12, 0, j * 12)))
            city_elements.append(district)

    return concat_shapes(*city_elements)