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