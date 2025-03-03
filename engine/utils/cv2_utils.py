import numpy as np
import cv2
import os
import imageio
from jaxtyping import Float


def cv2_downsize(f: Float[np.ndarray, "h w c"], downsize_factor: int | None = None) -> Float[np.ndarray, "h w c"]:
    if downsize_factor is None:
        return f
    return cv2.resize(f, (0, 0), fx=1 / downsize_factor, fy=1 / downsize_factor, interpolation=cv2.INTER_AREA)


def load_rgb_png(path: str, downsize_factor: int | None = None) -> Float[np.ndarray, "h w 3"]:
    f = imageio.imread(path)
    assert f.dtype == np.uint8, f.dtype
    assert len(f.shape) == 3 and f.shape[2] == 3, f.shape
    f = f.astype(np.float32) / 255
    f = cv2_downsize(f, downsize_factor)
    return f


def write_rgb_png(path: str, arr: Float[np.ndarray, "h w 3"]):
    assert len(arr.shape) == 3 and arr.dtype == np.float32 and arr.shape[2] == 3, (arr.dtype, arr.shape)
    assert os.path.exists(os.path.dirname(path)), path
    if arr.min() < 0 or arr.max() > 1:
        print(f'[WARNING] expected arr in [0, 1], got arr.min()={arr.min()}, arr.max()={arr.max()}')
    arr = (arr * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))