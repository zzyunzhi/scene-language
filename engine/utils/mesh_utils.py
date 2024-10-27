import numpy as np
import trimesh


def rewrite_mesh(save_path: str):
    mesh: trimesh.Trimesh = trimesh.load_mesh(save_path, process=False)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    mesh.export(save_path)
