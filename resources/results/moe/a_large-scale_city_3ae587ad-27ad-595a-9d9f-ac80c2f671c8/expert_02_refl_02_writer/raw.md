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

    # Add windows
    windows = []
    window_width = width * 0.15
    window_height = height * 0.05
    window_depth = 0.01
    window_color = (0.9, 0.9, 0.7)

    def window_loop_fn(i):
        row = i // 4
        col = i % 4
        window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
        x_pos = (col - 1.5) * (width * 0.22)
        y_pos = (row * height * 0.12) + (height * 0.1)
        z_pos = depth / 2 + window_depth/2  # Position windows on the surface
        return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))

    windows = loop(20, window_loop_fn)

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

@register("Creates a road segment")
def road_segment(length: float, width: float = 0.5) -> Shape:
    road = primitive_call('cube', shape_kwargs={'scale': (width, 0.01, length)}, color=(0.2, 0.2, 0.2))
    road = transform_shape(road, translation_matrix((0, 0.005, 0)))  # Position road slightly above ground

    # Road markings
    marking_width = width * 0.05
    marking_length = length * 0.1

    def marking_loop_fn(i):
        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, 0.02, marking_length)}, color=(1.0, 1.0, 1.0))
        z_pos = (i - 4) * length * 0.2
        return transform_shape(marking, translation_matrix((0, 0.01, z_pos)))  # Position markings slightly above road

    markings = loop(9, marking_loop_fn)

    return concat_shapes(road, markings)

@register("Creates a traffic light")
def traffic_light() -> Shape:
    # Pole
    pole = primitive_call('cylinder', shape_kwargs={'radius': 0.02, 'p0': (0, 0, 0), 'p1': (0, 0.5, 0)}, color=(0.3, 0.3, 0.3))
    
    # Light housing
    housing = primitive_call('cube', shape_kwargs={'scale': (0.1, 0.2, 0.1)}, color=(0.2, 0.2, 0.2))
    housing = transform_shape(housing, translation_matrix((0, 0.6, 0)))
    
    # Lights
    red_light = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(1.0, 0.0, 0.0))
    red_light = transform_shape(red_light, translation_matrix((0, 0.67, 0.05)))
    
    yellow_light = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(1.0, 1.0, 0.0))
    yellow_light = transform_shape(yellow_light, translation_matrix((0, 0.6, 0.05)))
    
    green_light = primitive_call('sphere', shape_kwargs={'radius': 0.03}, color=(0.0, 1.0, 0.0))
    green_light = transform_shape(green_light, translation_matrix((0, 0.53, 0.05)))
    
    return concat_shapes(pole, housing, red_light, yellow_light, green_light)

@register("Creates a street lamp")
def street_lamp() -> Shape:
    # Pole
    pole = primitive_call('cylinder', shape_kwargs={'radius': 0.02, 'p0': (0, 0, 0), 'p1': (0, 0.6, 0)}, color=(0.3, 0.3, 0.3))
    
    # Lamp head
    lamp_head = primitive_call('sphere', shape_kwargs={'radius': 0.05}, color=(1.0, 0.9, 0.6))
    lamp_head = transform_shape(lamp_head, translation_matrix((0, 0.65, 0)))
    
    return concat_shapes(pole, lamp_head)

@register("Creates a city block with buildings in a grid pattern")
def city_block(block_size: float, num_buildings: int) -> Shape:
    # Calculate grid dimensions
    grid_size = int(math.sqrt(num_buildings))
    if grid_size * grid_size < num_buildings:
        grid_size += 1
    
    def building_loop_fn(i):
        if i >= num_buildings:
            return []
        
        building_type = i % 3
        row = i // grid_size
        col = i % grid_size
        
        # Calculate position in grid
        spacing = block_size / (grid_size + 1)
        x_pos = (col - (grid_size-1)/2) * spacing
        z_pos = (row - (grid_size-1)/2) * spacing
        
        # Add slight randomness to position
        x_pos += math.sin(i * 0.5) * spacing * 0.1
        z_pos += math.cos(i * 0.5) * spacing * 0.1
        
        if building_type == 0:
            width = 0.3 + (i % 5) * 0.05
            height = 1.5 + (i % 7) * 0.3  # Make skyscrapers significantly taller
            depth = 0.3 + (i % 4) * 0.05
            building_shape = library_call('skyscraper', width=width, height=height, depth=depth)
        elif building_type == 1:
            width = 0.4 + (i % 3) * 0.05
            height = 0.6 + (i % 4) * 0.1
            depth = 0.4 + (i % 3) * 0.05
            building_shape = library_call('building', width=width, height=height, depth=depth)
        else:
            width = 0.3 + (i % 2) * 0.05
            height = 0.3 + (i % 3) * 0.05
            depth = 0.3 + (i % 2) * 0.05
            building_shape = library_call('house', width=width, height=height, depth=depth)
        
        return transform_shape(building_shape, translation_matrix((x_pos, 0, z_pos)))

    buildings = loop(grid_size * grid_size, building_loop_fn)
    
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
    
    # Add street lamps
    def lamp_loop_fn(i):
        pos = i / 8 * 2 * math.pi
        x = (block_size/2 - 0.1) * math.cos(pos)
        z = (block_size/2 - 0.1) * math.sin(pos)
        lamp = library_call('street_lamp')
        return transform_shape(lamp, translation_matrix((x, 0, z)))
    
    street_lamps = loop(8, lamp_loop_fn)
    
    street_furniture = concat_shapes(traffic_light1, traffic_light2, traffic_light3, traffic_light4, street_lamps)
    
    return concat_shapes(buildings, street_furniture)

@register("Creates a park with trees")
def park(size: float) -> Shape:
    # Base grass
    grass = primitive_call('cube', shape_kwargs={'scale': (size, 0.01, size)}, color=(0.2, 0.7, 0.2))
    grass = transform_shape(grass, translation_matrix((0, 0.005, 0)))  # Position grass slightly above ground

    # Trees
    def tree_loop_fn(i):
        row = i // 3
        col = i % 3

        # Tree trunk
        trunk_height = 0.2 + (i % 5) * 0.04  # More consistent tree heights
        trunk = primitive_call('cylinder', shape_kwargs={
            'radius': 0.03,
            'p0': (0, 0, 0),
            'p1': (0, trunk_height, 0)
        }, color=(0.5, 0.3, 0.1))

        # Tree top
        top = primitive_call('sphere', shape_kwargs={'radius': 0.1}, color=(0.1, 0.5, 0.1))
        top = transform_shape(top, translation_matrix((0, trunk_height + 0.05, 0)))

        tree = concat_shapes(trunk, top)

        x_pos = (col - 1) * (size / 3)
        z_pos = (row - 1) * (size / 3)

        # Add some randomness to positions
        x_pos += math.sin(i * 0.7) * 0.1
        z_pos += math.cos(i * 0.7) * 0.1

        return transform_shape(tree, translation_matrix((x_pos, 0, z_pos)))

    trees = loop(9, tree_loop_fn)

    # Add benches (smaller and more proportional)
    bench1 = primitive_call('cube', shape_kwargs={'scale': (0.2, 0.03, 0.06)}, color=(0.6, 0.4, 0.2))
    bench1 = transform_shape(bench1, translation_matrix((size/4, 0.03, size/4)))
    
    bench2 = primitive_call('cube', shape_kwargs={'scale': (0.2, 0.03, 0.06)}, color=(0.6, 0.4, 0.2))
    bench2 = transform_shape(bench2, translation_matrix((-size/4, 0.03, -size/4)))
    
    # Add a small fountain
    fountain_base = primitive_call('cylinder', shape_kwargs={'radius': 0.15, 'p0': (0, 0, 0), 'p1': (0, 0.02, 0)}, color=(0.7, 0.7, 0.7))
    fountain_water = primitive_call('cylinder', shape_kwargs={'radius': 0.12, 'p0': (0, 0.02, 0), 'p1': (0, 0.03, 0)}, color=(0.6, 0.8, 1.0))
    fountain = concat_shapes(fountain_base, fountain_water)
    fountain = transform_shape(fountain, translation_matrix((0, 0, 0)))

    return concat_shapes(grass, trees, bench1, bench2, fountain)

@register("Creates a connected road network for the city")
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
    
    return concat_shapes(*roads)

@register("Creates a large-scale city with multiple blocks and connected roads")
def city(size: int = 4) -> Shape:
    block_size = 2.0
    road_width = 0.5
    block_spacing = road_width  # Space between blocks equals road width
    
    # Create ground
    ground_size = size * (block_size + block_spacing) + block_spacing
    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, 0.1, ground_size)}, color=(0.6, 0.6, 0.6))
    ground = transform_shape(ground, translation_matrix((0, -0.05, 0)))  # Position ground so its top is at y=0
    
    # Create connected road network
    roads = library_call('road_network', size=size, block_size=block_size, spacing=block_spacing)
    
    # Create city blocks
    def city_block_loop_fn(i):
        row = i // size
        col = i % size
        
        # Determine block type based on position
        # More skyscrapers in the center, more residential on the outskirts
        distance_from_center = math.sqrt((row - size/2 + 0.5)**2 + (col - size/2 + 0.5)**2)
        block_type = 2 if distance_from_center > size/2 else (0 if distance_from_center < size/4 else 1)
        
        if block_type < 2:
            # Regular city block - more buildings in central areas
            num_buildings = 12 if block_type == 0 else 9
            block = library_call('city_block', block_size=block_size, num_buildings=num_buildings)
        else:
            # Park
            block = library_call('park', size=block_size)
        
        x_pos = (col - size/2 + 0.5) * (block_size + block_spacing)
        z_pos = (row - size/2 + 0.5) * (block_size + block_spacing)
        
        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
    
    city_blocks = loop(size * size, city_block_loop_fn)
    
    return concat_shapes(ground, roads, city_blocks)
```