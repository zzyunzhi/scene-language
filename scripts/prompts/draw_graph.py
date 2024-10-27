from typing import Union, Callable, Set, List
import numpy as np
from type_utils import Shape
from shape_utils import concat_shapes, transform_shape, placeholder
from math_utils import rotation_matrix, translation_matrix
from graphviz import Digraph
from jaxtyping import Float


# conceptually it's a computation graph; a Hole (like a vector) instance is a node
# a Hole can be instantiated multiple times, into multiple nodes
# nodes can be connected with `+` operator
# it can be transformed with `*` operator with an instance from Transform (like a scalar)
# Hole.implementation specifies the computation flow where holes are transformed and combined into shapes via `+` and `*`

"""
Let's define our basic elements:
M: the set of all possible Entity objects (Manifold)
G: the set of all possible Transformation objects (LieGroup)
H: the set of all possible Hole objects

Now, let's define the operations:
Sum operation on Manifolds:
⊕ : M × M → M
For a, b ∈ M, we write: a ⊕ b = c, where c ∈ M
Group action of LieGroup on Manifold:
• : G × M → M
For g ∈ G and m ∈ M, we write: g • m = n, where n ∈ M
Composition of LieGroup elements:
 : G × G → G
For g, h ∈ G, we write: g ∘ h = k, where k ∈ G
4. Hole implementation function:
f : H → (Args → M)
Where Args is the set of possible arguments for a Hole


Now, we can write out the main operations in your code as equations:
Summing two Manifolds (previously Entity.sum):
m1, m2 ∈ M
m1 ⊕ m2 = concat_shapes(m1, m2)
2. LieGroup acting on a Manifold (previously Entity.act and Transformation.act):
g ∈ G, m ∈ M
g • m = transform_shape(m, g)
Composing two LieGroup elements (previously Transformation.compose):
g1, g2 ∈ G
g1 ∘ g2 = g1.data @ g2.data (where @ is matrix multiplication)
Implementing a Hole:
h ∈ H
f(h) : Args → M
These equations describe the basic structure and operations of your system without using overly complex mathematical terminology. They capture the essence of how Manifolds can be combined, how LieGroup elements act on Manifolds and compose with each other, and how Holes are implemented as functions that produce Manifolds.

"""


class Hole:
    def __init__(self, name: str, docstring: str = '') -> None:
        """
        Args:
            name: str - a unique name
            docstring: str - a detailed function docstring
        """
        self.name: str = name
        self.docstring: str = docstring
        self.fn: Callable[..., Entity] = None  # type: ignore

    @property
    def is_implemented(self) -> bool:
        return self.fn is not None

    def implement(self, fn: Callable[..., 'Entity']):
        self.fn = fn

    def __repr__(self):
        return f'Hole({self.name})'

    def __call__(self, **kwargs) -> 'Entity':
        assert self.is_implemented, self.name
        entity = self.fn(**kwargs)
        entity.cls = self
        entity.embd = kwargs
        return entity


class Entity:
    def __init__(self, data: Shape):
        """
        Args:
            data: Shape - a shape
        """
        self.data: Shape = data
        self.cls: Hole = None  # type: ignore # may be set later
        self.embd: dict = None  # type: ignore # may be set later
        self._prev: List['Entity'] = []
        self._op: str = ''

    def sum(self, other: 'Entity') -> 'Entity':
        result = Entity(concat_shapes(self.data, other.data))
        result._prev = [self, other]
        result._op = 'sum'
        return result


class Transformation:
    def __init__(self, data: Float[np.ndarray, '4 4']):
        """
        Args:
            data - a 4x4 transformation matrix
        """
        assert data.shape == (4, 4), f'expected a 4x4 matrix, got {data.shape}'
        self.data = data

    def compose(self, other: 'Transformation') -> 'Transformation':
        return Transformation(self.data @ other.data)

    def act(self, entity: Entity) -> Entity:
        result = Entity(transform_shape(entity.data, self.data))
        result._prev = [entity]
        result._op = 'transform'
        return result



def test_draw():
    apple_cls = Hole('apple', 'a green apple')
    man_cls = Hole('man', 'a man')
    painting_cls = Hole('painting', 'a painting')

    def _man_cls_fn():
        return Entity(placeholder(center=(0, 0, 0), scale=(.5, 1, .5), color=(0.2, 0.2, 0.2)))

    man_cls.fn = _man_cls_fn

    def _apple_cls_fn(color: str):
        return Entity(placeholder(center=(0, 0, 0), scale=(.1, .1, .1), color=(color, 0, 0)))

    apple_cls.fn = _apple_cls_fn

    def _painting_cls_fn(apple_color: str):
        # Son of Man, by René Magritte
        man = man_cls()
        apple = apple_cls(color=apple_color)
        man_pose = Transformation(translation_matrix(np.array([0, 0, 0])) @ rotation_matrix(np.pi / 2, np.array([1, 0, 0]), point=np.array([0, 0, 0])))
        apple_pose = Transformation(translation_matrix(np.array([0.5, 0, 0])))
        man = man_pose.act(man)
        apple = apple_pose.act(apple)
        return man.sum(apple)
        
    painting_cls.implement(_painting_cls_fn)

    # now we've defined the computation patterns... time to instantiate and execute a computation graph

    painting = painting_cls(apple_color='green')
    print(painting.data)

    # Visualize the computation graph
    dot = draw_dot(painting)
    dot.render('computation_graph', view=True)


def trace(root: Entity):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root: Entity, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={
        'rankdir': rankdir,
        'bgcolor': 'transparent',
    })
    
    for n in nodes:
        label = f"{n.cls.name if n.cls else 'Entity'}"

        dot.node(name=str(id(n)), label=label, shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot


if __name__ == '__main__':
    test_draw()
