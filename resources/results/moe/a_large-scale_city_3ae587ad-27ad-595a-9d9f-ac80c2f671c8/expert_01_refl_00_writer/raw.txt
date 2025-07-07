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
    window_height = height * 0.05
    window_depth = 0.01
    window_color = (0.9, 0.9, 0.7)
    
    # Calculate number of windows per side
    windows_per_row = max(int(width / (window_width * 1.5)), 2)
    windows_per_column = max(int(height / (window_height * 1.5)), 4)
    
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
    
    # Roof
    roof_height = height * 0.5
    roof_color = (0.6, 0.3, 0.2)
    
    # Create a triangular roof using a scaled and rotated cube
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=roof_color)
    roof_center = compute_shape_center(roof)
    
    # Transform to make a triangular roof
    roof = transform_shape(roof, rotation_matrix(math.pi/4, (0, 0, 1), roof_center))
    roof = transform_shape(roof, scale_matrix(0.7, roof_center))
    
    # Position the roof on top of the house
    house_top = compute_shape_max(main_house)[1]  # y-coordinate of the top of the house
    roof_bottom = compute_shape_min(roof)[1]  # y-coordinate of the bottom of the roof
    roof = transform_shape(roof, translation_matrix((0, house_top - roof_bottom + height * 0.1, 0)))
    
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
    
    return concat_shapes(main_house, roof, door, window1, window2)

@register("Creates a road segment")
def road(length: float, width: float = 1.0, color: tuple = (0.2, 0.2, 0.2)) -> Shape:
    road_height = 0.05
    road_shape = primitive_call('cube', shape_kwargs={'scale': (width, road_height, length)}, color=color)
    
    # Add road markings
    marking_width = width * 0.05
    marking_length = length * 0.1
    marking_color = (1.0, 1.0, 1.0)
    
    def marking_loop_fn(i):
        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, road_height * 1.01, marking_length)}, color=marking_color)
        z_pos = (i - 4.5) * (length / 10)
        return transform_shape(marking, translation_matrix((0, 0, z_pos)))
    
    markings = loop(10, marking_loop_fn)
    
    return concat_shapes(road_shape, markings)

@register("Creates a park with trees")
def park(width: float, depth: float) -> Shape:
    # Ground
    ground = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.2, 0.6, 0.2))
    
    def tree_loop_fn(i):
        # Calculate position
        row = i % 5
        col = i // 5
        x_pos = (row - 2) * (width / 5)
        z_pos = (col - 2) * (depth / 5)
        
        # Add some randomness to positions
        x_pos += np.random.uniform(-width/12, width/12)
        z_pos += np.random.uniform(-depth/12, depth/12)
        
        # Create tree trunk
        trunk_height = np.random.uniform(0.3, 0.6)
        trunk_radius = np.random.uniform(0.05, 0.1)
        trunk = primitive_call('cylinder', 
                              shape_kwargs={'radius': trunk_radius, 
                                           'p0': (x_pos, 0.05, z_pos), 
                                           'p1': (x_pos, trunk_height, z_pos)}, 
                              color=(0.5, 0.3, 0.1))
        
        # Create tree foliage
        foliage_radius = np.random.uniform(0.3, 0.5)
        foliage = primitive_call('sphere', 
                                shape_kwargs={'radius': foliage_radius}, 
                                color=(0.0, np.random.uniform(0.5, 0.8), 0.0))
        foliage = transform_shape(foliage, translation_matrix((x_pos, trunk_height + foliage_radius * 0.7, z_pos)))
        
        return concat_shapes(trunk, foliage)
    
    trees = loop(25, tree_loop_fn)
    
    return concat_shapes(ground, trees)

@register("Creates a city block with buildings")
def city_block(width: float, depth: float) -> Shape:
    # Create the base
    base = primitive_call('cube', shape_kwargs={'scale': (width, 0.1, depth)}, color=(0.6, 0.6, 0.6))
    
    def building_loop_fn(i):
        # Calculate position
        row = i % 3
        col = i // 3
        x_pos = (row - 1) * (width / 3)
        z_pos = (col - 1) * (depth / 3)
        
        # Add some randomness to positions
        x_pos += np.random.uniform(-width/12, width/12)
        z_pos += np.random.uniform(-depth/12, depth/12)
        
        # Randomize building properties
        building_width = np.random.uniform(width/6, width/4)
        building_depth = np.random.uniform(depth/6, depth/4)
        building_height = np.random.uniform(1.0, 3.0)
        
        # Randomly choose between different building types
        building_type = np.random.choice(['skyscraper', 'building', 'house'])
        
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
        
        # Position the building
        building_bottom = compute_shape_min(building_shape)[1]
        base_top = compute_shape_max(base)[1]
        building_shape = transform_shape(building_shape, translation_matrix((x_pos, base_top - building_bottom, z_pos)))
        
        return building_shape
    
    buildings = loop(9, building_loop_fn)
    
    return concat_shapes(base, buildings)

@register("Creates a city grid with blocks and roads")
def city_grid(size: int = 5, block_size: float = 4.0, road_width: float = 1.0) -> Shape:
    city = []
    
    # Create city blocks
    def block_loop_fn(i):
        row = i % size
        col = i // size
        
        # Calculate position with roads in between
        total_unit_size = block_size + road_width
        x_pos = (row - (size-1)/2) * total_unit_size
        z_pos = (col - (size-1)/2) * total_unit_size
        
        # Randomly choose between regular block and park
        if np.random.random() < 0.8:  # 80% chance of a building block
            block = library_call('city_block', width=block_size, depth=block_size)
        else:  # 20% chance of a park
            block = library_call('park', width=block_size, depth=block_size)
        
        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
    
    blocks = loop(size * size, block_loop_fn)
    city.append(blocks)
    
    # Create horizontal roads
    def h_road_loop_fn(i):
        row = i % (size + 1)
        col = i // (size + 1)
        
        if col < size:  # Only create roads between blocks
            total_unit_size = block_size + road_width
            x_pos = (row - size/2) * total_unit_size - road_width/2
            z_pos = (col - (size-1)/2) * total_unit_size
            
            road = library_call('road', length=block_size, width=road_width)
            road = transform_shape(road, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
            return transform_shape(road, translation_matrix((x_pos, 0, z_pos)))
        return []
    
    h_roads = loop((size + 1) * size, h_road_loop_fn)
    city.append(h_roads)
    
    # Create vertical roads
    def v_road_loop_fn(i):
        row = i % size
        col = i // size
        
        if col < (size + 1):  # Create roads between and around blocks
            total_unit_size = block_size + road_width
            x_pos = (row - (size-1)/2) * total_unit_size
            z_pos = (col - size/2) * total_unit_size - road_width/2
            
            road = library_call('road', length=block_size, width=road_width)
            return transform_shape(road, translation_matrix((x_pos, 0, z_pos)))
        return []
    
    v_roads = loop(size * (size + 1), v_road_loop_fn)
    city.append(v_roads)
    
    return concat_shapes(*city)

@register("Creates a complete city with downtown and suburbs")
def large_scale_city() -> Shape:
    # Create downtown area with tall buildings
    downtown = library_call('city_grid', size=3, block_size=5.0, road_width=1.2)
    
    # Create suburban areas
    suburb1 = library_call('city_grid', size=2, block_size=6.0, road_width=1.0)
    suburb1 = transform_shape(suburb1, translation_matrix((15, 0, 15)))
    
    suburb2 = library_call('city_grid', size=2, block_size=6.0, road_width=1.0)
    suburb2 = transform_shape(suburb2, translation_matrix((-15, 0, 15)))
    
    suburb3 = library_call('city_grid', size=2, block_size=6.0, road_width=1.0)
    suburb3 = transform_shape(suburb3, translation_matrix((15, 0, -15)))
    
    suburb4 = library_call('city_grid', size=2, block_size=6.0, road_width=1.0)
    suburb4 = transform_shape(suburb4, translation_matrix((-15, 0, -15)))
    
    # Create connecting roads
    road1 = library_call('road', length=7.0, width=1.5)
    road1 = transform_shape(road1, translation_matrix((8, 0, 0)))
    
    road2 = library_call('road', length=7.0, width=1.5)
    road2 = transform_shape(road2, translation_matrix((-8, 0, 0)))
    
    road3 = library_call('road', length=7.0, width=1.5)
    road3 = transform_shape(road3, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road3 = transform_shape(road3, translation_matrix((0, 0, 8)))
    
    road4 = library_call('road', length=7.0, width=1.5)
    road4 = transform_shape(road4, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road4 = transform_shape(road4, translation_matrix((0, 0, -8)))
    
    # Create ground
    ground = primitive_call('cube', shape_kwargs={'scale': (50, 0.1, 50)}, color=(0.3, 0.3, 0.3))
    ground = transform_shape(ground, translation_matrix((0, -0.1, 0)))
    
    return concat_shapes(ground, downtown, suburb1, suburb2, suburb3, suburb4, road1, road2, road3, road4)
```