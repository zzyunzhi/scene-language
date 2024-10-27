Here are some examples of how to use `helper.py`:
```python
from helper import *

"""
A pile of books on a desk
"""


@register()
def book(scale: P) -> Shape:
    return primitive_call('cube', color=(.6, .3, .1), shape_kwargs={'scale': scale})


@register()
def books(width: float, length: float, book_height: float, num_books: int) -> Shape:
    def loop_fn(i) -> Shape:
        book_shape = library_call('book', scale=(width, book_height, length))
        book_shape = transform_shape(book_shape, translation_matrix([np.random.uniform(-0.05, 0.05), i * book_height, np.random.uniform(-0.05, 0.05)]))  # FIRST translate
        book_center = compute_shape_center(book_shape)  # must be computed AFTER transformation!!
        return transform_shape(book_shape, rotation_matrix(np.random.uniform(-0.1, 0.1), direction=(0, 1, 0), point=book_center))  # THEN tilt

    return loop(num_books, loop_fn)


@register()
def desk(scale: P) -> Shape:
    return primitive_call('cube', color=(.4, .2, .1), shape_kwargs={'scale': scale})


@register()
def desk_with_books() -> Shape:
    desk_shape = library_call('desk', scale=(1, .1, .5))
    books_shape = library_call('books', width=.21, length=.29, book_height=.05, num_books=3)
    _, desk_top, _ = compute_shape_max(desk_shape)
    _, books_bottom, _ = compute_shape_min(books_shape)
    return concat_shapes(
        desk_shape,
        transform_shape(books_shape, translation_matrix((0, desk_top - books_bottom, 0)))  # stack books on top of desk
    )

```
IMPORTANT: THE FUNCTIONS ABOVE ARE JUST EXAMPLES, YOU CANNOT USE THEM IN YOUR PROGRAM! 

Now, write a similar program for the given task:    
```python
from helper import *

"""
a chessboard with a full set of chess pieces
"""
```
