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