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
    
    def window_loop_fn(i):
        row = i // 4
        col = i % 4
        window = primitive_call('cube', shape_kwargs={'scale': (window_width, window_height, window_depth)}, color=window_color)
        x_pos = (col - 1.5) * (width * 0.22)
        y_pos = (row * height * 0.12) + (height * 0.1)
        z_pos = depth / 2
        return transform_shape(window, translation_matrix((x_pos, y_pos, z_pos)))
    
    windows = loop(20, window_loop_fn)
    
    return concat_shapes(main_building, windows)

@register("Creates a house with a roof")
def house(width: float, height: float, depth: float) -> Shape:
    # Main house
    main_house = primitive_call('cube', shape_kwargs={'scale': (width, height, depth)}, color=(0.8, 0.6, 0.5))
    
    # Roof
    roof_height = height * 0.5
    roof = primitive_call('cube', shape_kwargs={'scale': (width * 1.2, roof_height, depth * 1.2)}, color=(0.6, 0.3, 0.2))
    roof = transform_shape(roof, translation_matrix((0, height / 2 + roof_height / 2, 0)))
    
    # Door
    door_width = width * 0.2
    door_height = height * 0.4
    door = primitive_call('cube', shape_kwargs={'scale': (door_width, door_height, 0.01)}, color=(0.4, 0.2, 0.1))
    door = transform_shape(door, translation_matrix((0, -height / 2 + door_height / 2, depth / 2)))
    
    # Windows
    window1 = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, height * 0.15, 0.01)}, color=(0.8, 0.8, 1.0))
    window1 = transform_shape(window1, translation_matrix((width * 0.25, 0, depth / 2)))
    
    window2 = primitive_call('cube', shape_kwargs={'scale': (width * 0.15, height * 0.15, 0.01)}, color=(0.8, 0.8, 1.0))
    window2 = transform_shape(window2, translation_matrix((-width * 0.25, 0, depth / 2)))
    
    return concat_shapes(main_house, roof, door, window1, window2)

@register("Creates a road segment")
def road_segment(length: float, width: float = 0.5) -> Shape:
    road = primitive_call('cube', shape_kwargs={'scale': (width, 0.01, length)}, color=(0.2, 0.2, 0.2))
    
    # Road markings
    marking_width = width * 0.05
    marking_length = length * 0.1
    
    def marking_loop_fn(i):
        marking = primitive_call('cube', shape_kwargs={'scale': (marking_width, 0.02, marking_length)}, color=(1.0, 1.0, 1.0))
        z_pos = (i - 4) * length * 0.2
        return transform_shape(marking, translation_matrix((0, 0, z_pos)))
    
    markings = loop(9, marking_loop_fn)
    
    return concat_shapes(road, markings)

@register("Creates a city block with buildings")
def city_block(block_size: float, num_buildings: int) -> Shape:
    def building_loop_fn(i):
        building_type = i % 3
        position_angle = (i / num_buildings) * 2 * math.pi
        radius = block_size * 0.4
        
        x_pos = radius * math.cos(position_angle)
        z_pos = radius * math.sin(position_angle)
        
        if building_type == 0:
            width = np.random.uniform(0.3, 0.5)
            height = np.random.uniform(0.8, 1.5)
            depth = np.random.uniform(0.3, 0.5)
            building_shape = library_call('skyscraper', width=width, height=height, depth=depth)
        elif building_type == 1:
            width = np.random.uniform(0.4, 0.6)
            height = np.random.uniform(0.5, 0.8)
            depth = np.random.uniform(0.4, 0.6)
            building_shape = library_call('building', width=width, height=height, depth=depth)
        else:
            width = np.random.uniform(0.3, 0.4)
            height = np.random.uniform(0.3, 0.5)
            depth = np.random.uniform(0.3, 0.4)
            building_shape = library_call('house', width=width, height=height, depth=depth)
        
        return transform_shape(building_shape, translation_matrix((x_pos, 0, z_pos)))
    
    buildings = loop(num_buildings, building_loop_fn)
    
    # Add roads around the block
    road1 = library_call('road_segment', length=block_size, width=0.5)
    road1 = transform_shape(road1, translation_matrix((0, -0.05, -block_size/2 - 0.25)))
    
    road2 = library_call('road_segment', length=block_size, width=0.5)
    road2 = transform_shape(road2, translation_matrix((0, -0.05, block_size/2 + 0.25)))
    
    road3 = library_call('road_segment', length=block_size, width=0.5)
    road3 = transform_shape(road3, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road3 = transform_shape(road3, translation_matrix((-block_size/2 - 0.25, -0.05, 0)))
    
    road4 = library_call('road_segment', length=block_size, width=0.5)
    road4 = transform_shape(road4, rotation_matrix(math.pi/2, (0, 1, 0), (0, 0, 0)))
    road4 = transform_shape(road4, translation_matrix((block_size/2 + 0.25, -0.05, 0)))
    
    return concat_shapes(buildings, road1, road2, road3, road4)

@register("Creates a park with trees")
def park(size: float) -> Shape:
    # Base grass
    grass = primitive_call('cube', shape_kwargs={'scale': (size, 0.01, size)}, color=(0.2, 0.7, 0.2))
    
    # Trees
    def tree_loop_fn(i):
        row = i // 3
        col = i % 3
        
        # Tree trunk
        trunk_height = np.random.uniform(0.2, 0.4)
        trunk = primitive_call('cylinder', shape_kwargs={
            'radius': 0.05,
            'p0': (0, 0, 0),
            'p1': (0, trunk_height, 0)
        }, color=(0.5, 0.3, 0.1))
        
        # Tree top
        top = primitive_call('sphere', shape_kwargs={'radius': 0.15}, color=(0.1, 0.5, 0.1))
        top = transform_shape(top, translation_matrix((0, trunk_height + 0.1, 0)))
        
        tree = concat_shapes(trunk, top)
        
        x_pos = (col - 1) * (size / 3)
        z_pos = (row - 1) * (size / 3)
        
        # Add some randomness to positions
        x_pos += np.random.uniform(-0.1, 0.1)
        z_pos += np.random.uniform(-0.1, 0.1)
        
        return transform_shape(tree, translation_matrix((x_pos, 0, z_pos)))
    
    trees = loop(9, tree_loop_fn)
    
    # Add a bench
    bench = primitive_call('cube', shape_kwargs={'scale': (0.4, 0.05, 0.1)}, color=(0.6, 0.4, 0.2))
    bench = transform_shape(bench, translation_matrix((size/4, 0.1, size/4)))
    
    return concat_shapes(grass, trees, bench)

@register("Creates a large-scale city with multiple blocks")
def city(size: int = 4) -> Shape:
    def city_block_loop_fn(i):
        row = i // size
        col = i % size
        
        block_type = (row + col) % 3
        
        if block_type < 2:
            # Regular city block
            block = library_call('city_block', block_size=2.0, num_buildings=6 + (row + col) % 4)
        else:
            # Park
            block = library_call('park', size=2.0)
        
        x_pos = (col - size/2 + 0.5) * 3.0
        z_pos = (row - size/2 + 0.5) * 3.0
        
        return transform_shape(block, translation_matrix((x_pos, 0, z_pos)))
    
    city_blocks = loop(size * size, city_block_loop_fn)
    
    # Create ground
    ground_size = size * 3.0 + 2.0
    ground = primitive_call('cube', shape_kwargs={'scale': (ground_size, 0.1, ground_size)}, color=(0.6, 0.6, 0.6))
    ground = transform_shape(ground, translation_matrix((0, -0.1, 0)))
    
    return concat_shapes(city_blocks, ground)
```