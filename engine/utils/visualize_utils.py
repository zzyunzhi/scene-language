from jaxtyping import Bool, Float
from typing import Optional
import numpy as np
from PIL import Image


def visualize_depth_map(depth_map: Float[np.ndarray, "h w"],
                        segm_map: Bool[np.ndarray, "h w"],
                        depth_min: Optional[float] = None,
                        depth_max: Optional[float] = None,
                        return_depth_min_max: bool = False,
) -> Float[np.ndarray, "h w"]:
    segm_map = segm_map.astype(bool)
    depth_values = depth_map[segm_map]
    depth_values = depth_values[depth_values != np.inf]
    if depth_min is None:
        depth_min = depth_values.min()  # np.percentile(depth_values, 0)
    if depth_max is None:
        depth_max = depth_values.max()  # np.percentile(depth_values, 100)
    if depth_max - depth_min < 1e-3:
        return np.zeros_like(depth_map, dtype=np.uint8)

    depth_map = ((depth_map - depth_min) / (depth_max - depth_min)).clip(0, 1)
    if return_depth_min_max:
        return depth_map, (depth_min, depth_max)
    return depth_map
