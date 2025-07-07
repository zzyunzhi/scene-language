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
                z_pos = depth / 2 + window_depth / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 1:  # Back
                x_pos = (row - (windows_per_row - 1) / 2) * (width / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                z_pos = -depth / 2 - window_depth / 2
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 2:  # Left
                z_pos = (row - (windows_per_row - 1) / 2) * (depth / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                x_pos = -width / 2 - window_depth / 2
                window = transform_shape(window, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
            elif side == 3:  # Right
                z_pos = (row - (windows_per_row - 1) / 2) * (depth / windows_per_row)
                y_pos = (col - (windows_per_column - 1) / 2) * (height / windows_per_column)
                x_pos = width / 2 + window_depth / 2
                window = transform_shape(window, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
                window = transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

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

    return concat_shapes(road_shape, markings)

@register("Creates a park with trees and paths")
def park(width: float, depth: float) -> Shape:
    # Ground
    ground = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))

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

        # Create tree trunk - ensure it starts at the ground level
        trunk_height = np.random.uniform(0.3, 0.6)
        trunk_radius = np.random.uniform(0.05, 0.1)
        trunk = primitive_call('cylinder',
                              shape_kwargs={'radius': trunk_radius,
                                           'p0': (x_pos, 0.1, z_pos),  # Start at ground level
                                           'p1': (x_pos, 0.1 + trunk_height, z_pos)},
                              color=(0.5, 0.3, 0.1))

        # Create tree foliage
        foliage_radius = np.random.uniform(0.3, 0.5)
        foliage = primitive_call('sphere',
                                shape_kwargs={'radius': foliage_radius},
                                color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
        foliage = transform_shape(foliage, translation_matrix((x_pos, 0.1 + trunk_height + foliage_radius * 0.7, z_pos)))

        # Add a bench near some trees (randomly)
        if np.random.random() < 0.2:  # 20% chance of adding a bench
            bench_seat = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.05, 0.2)}, color=(0.6, 0.4, 0.2))
            bench_back = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.2, 0.05)}, color=(0.6, 0.4, 0.2))

            bench_seat = transform_shape(bench_seat, translation_matrix((x_pos + 0.3, 0.15, z_pos)))
            bench_back = transform_shape(bench_back, translation_matrix((x_pos + 0.3, 0.25, z_pos - 0.075)))

            return concat_shapes(trunk, foliage, bench_seat, bench_back)

        return concat_shapes(trunk, foliage)

    trees = loop(25, tree_loop_fn)

    return concat_shapes(ground, h_path, v_path, center_area, trees)

@register("Creates a traffic light")
def traffic_light() -> Shape:
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

    return concat_shapes(pole, housing, red_light, yellow_light, green_light)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float, is_downtown: bool = False) -> Shape:
    # Create the base
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.6, 0.6, 0.6))

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

        if building_type == 'skyscraper':
            building_shape = library_call('skyscraper',
                                         width=building_width,
                                         height=building_height * 2,
                                         depth=building_depth,
                                         color=(np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6), np.random.uniform(0.5, 0.7)))
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

    return concat_shapes(base, buildings)

@register("Creates a city intersection with traffic lights")
def intersection(road_width: float = 1.0) -> Shape:
    # Create the intersection base
    intersection_base = primitive_call('cube',
                                      shape_kwargs={'scale': (road_width * 2, 0.05, road_width * 2)},
                                      color=(0.2, 0.2, 0.2))

    # Add traffic lights at the corners
    traffic_light1 = library_call('traffic_light')
    traffic_light1 = transform_shape(traffic_light1,
                                    translation_matrix((road_width * 0.8, 0, road_width * 0.8)))

    traffic_light2 = library_call('traffic_light')
    traffic_light2 = transform_shape(traffic_light2,
                                    translation_matrix((-road_width * 0.8, 0, road_width * 0.8)))

    traffic_light3 = library_call('traffic_light')
    traffic_light3 = transform_shape(traffic_light3,
                                    translation_matrix((road_width * 0.8, 0, -road_width * 0.8)))

    traffic_light4 = library_call('traffic_light')
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

    return concat_shapes(intersection_base, traffic_light1, traffic_light2,
                         traffic_light3, traffic_light4, crosswalk_markings)

@register("Creates a city grid with blocks and roads")
def city_grid(size: int = 5, block_size: float = 4.0, road_width: float = 1.0, is_downtown: bool = False) -> Shape:
    city = []

    # Calculate total unit size (block + road)
    total_unit_size = block_size + road_width

    # Create city blocks
    def block_loop_fn(i):
        row = i % size
        col = i // size

        # Calculate position with roads in between
        x_pos = (row - (size-1)/2) * total_unit_size
        z_pos = (col - (size-1)/2) * total_unit_size

        # Randomly choose between regular block and park
        if is_downtown:
            park_chance = 0.1  # Less parks downtown
        else:
            park_chance = 0.2  # More parks in suburbs

        if np.random.random() < park_chance:
            block = library_call('park', width=block_size, depth=block_size)
        else:
            block = library_call('city_block', width=block_size, depth=block_size, is_downtown=is_downtown)

        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))

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

    return concat_shapes(*city)

@register("Creates a parking lot")
def parking_lot(width: float, depth: float) -> Shape:
    # Base
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.05, depth)}, color=(0.3, 0.3, 0.3))

    # Parking spaces
    space_width = 0.8
    space_depth = 1.6
    spacing = 0.1

    def parking_space_fn(i):
        rows_per_side = int((depth - 2) / (space_depth + spacing))
        spaces_per_row = int((width - 2) / (space_width + spacing))

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
    rows_per_side = int((depth - 2) / (space_depth + spacing))
    spaces_per_row = int((width - 2) / (space_width + spacing))
    total_spaces = rows_per_side * spaces_per_row * 2

    parking_spaces = loop(total_spaces, parking_space_fn)

    # Central driving lane
    lane = primitive_call('cube', shape_kwargs={'scale': (width - 2, 0.06, depth - 2 - 2*space_depth - 2*spacing)}, color=(0.2, 0.2, 0.2))

    return concat_shapes(base, parking_spaces, lane)

@register("Creates a complete city with downtown and suburbs")
def large_scale_city() -> Shape:
    # Create ground
    ground = primitive_call('cube', shape_kwargs={'scale': (50, 0.1, 50)}, color=(0.3, 0.3, 0.3))
    ground = transform_shape(ground, translation_matrix((0, -0.05, 0)))  # Position at y=-0.05 so top is at y=0

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

    return concat_shapes(ground, downtown, suburb1, suburb2, suburb3, suburb4,
                         road1, road2, road3, road4, parking1, parking2)
```

Provide your critiques and suggestions for improvement below in a formatted list.


![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/expert_01_refl_02_writer/renderings/exposed_large_scale_city_rover_background_frame_00/rendering_traj_000.png)



![image](/Users/yzzhang/dev/scene-language/scripts/outputs/run_self_reflect_with_moe_20250707-111552_a6431c8b-82dc-4621-b943-fa0268d47678/a_large-scale_city_3ae587ad-27ad-595a-9d9f-ac80c2f671c8/expert_01_refl_02_writer/renderings/exposed_large_scale_city_rover_background_frame_00/rendering_traj_003.png)

