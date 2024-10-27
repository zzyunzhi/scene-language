import torch
from _typeshed import Incomplete
from math_utils import rotation_matrix as rotation_matrix, translation_matrix as translation_matrix

device: Incomplete
SPP: int

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
def render_scene_w_pose(scene, rotation_rep: str = 'axis-aligned', trans: Incomplete | None = None, rot_x: Incomplete | None = None, rot_y: Incomplete | None = None, rot_z: Incomplete | None = None, mtx: Incomplete | None = None, cam: Incomplete | None = None, spp: int = 256, seed: int = 1): ...
def sds_guidance(img, model_prompt_processor, model_guidance, sample_cam): ...
def debug_layout_optimize(scene, keys): ...
def compute_closest_fov(camera_location, box): ...
def layout_optimize_mi(shape, prompt, save_dir, preset_xml_path, target_box, optimize_iter: float = 100.0): ...
def layout_optimize(shape, prompt, save_dir, preset_xml_path, target_box, optimize_iter: float = 100.0): ...
def random_sample_camera(batch_size: int = 1, elevation_range=[10, 50], azimuth_range=[-10, 150], camera_distance_range=[1.5, 2.0]): ...
