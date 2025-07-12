import kubric as kb
from kubric.simulator.pybullet import PyBullet as KubricSimulator
from typing import Any, Dict, List
from engine.constants import ENGINE_MODE, PROJ_DIR
from pathlib import Path
import sys
prompts_root = Path(PROJ_DIR) / 'scripts/prompts'
sys.path.append(prompts_root.as_posix())
from scripts.prompts.sketch_helper import parse_program
from engine.utils.graph_utils import get_root
from scripts.prompts.dsl_utils import set_seed
from pathlib import Path
import numpy as np
from scripts.prompts.helper import *
from scripts.prompts.impl_preset import core
from scripts.prompts.mi_helper import execute_from_preset
import pyquaternion
import re
from transforms3d._gohlketransforms import decompose_matrix

from abc import ABC, abstractmethod


# Adapted from https://github.com/openai/gym/blob/master/gym/core.py
class Env(ABC):
    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def test_kubric():
    scene = kb.Scene(
        gravity=(0, -10, 0),
        frame_end=24,
    )
    simulator = KubricSimulator(scene)
    cube = kb.Cube(
        name='box',
        position=[0, 0, 0],
    )
    scene.add(cube)
    simulator.run()
    np.testing.assert_allclose(cube.position[1], -0.5 * 10, atol=0.1)
    print('Pybullet simulator test passed!')


PROG_PATH = "resources/assets/jenga.py"

class PyBulletEnv(Env):
    def __init__(self):
        super().__init__()
        self.shape_scene: Shape | None = None
        self.shape_leaves: List[Shape] = []
        
        self.kb_scene: kb.Scene | None = None
        self.kb_engine: KubricSimulator | None = None
    
    def reset(self):
        library, library_equiv, _ = parse_program(['resources/examples/jenga.py'], roots=None)
        scene_name = get_root(library_equiv)
        with set_seed(0):
            shape = library[scene_name]['__target__']()

        # manually add floor
        floor_x, _, floor_z = compute_shape_center(shape)
        _, floor_y, _ = compute_shape_min(shape)
        shape = concat_shapes(shape, transform_shape(primitive_call('cube', color=(0.2, 0.2, 0.2), shape_kwargs={'scale': (5, 0.1, 5)}), translation_matrix((floor_x, floor_y - .05, floor_z))))

        self.shape_scene = shape
        self.shape_leaves = []
        self.kb_scene = kb.Scene(gravity=(0, -9.81, 0), step_rate=120)
        self.kb_engine = KubricSimulator(self.kb_scene)
        for eid in range(len(self.shape_scene)):
            if eid in [4, 5]:
                continue  # FIXME HARDCODED
            shape_leaf = self.shape_scene[eid:eid+1]
            rot = shape_leaf[0]['to_world'][:3, :3]
            # scale = np.linalg.norm(rot, axis=0)
            scale, _, _, translate, _ = decompose_matrix(shape_leaf[0]['to_world'])  # should be the same
            rot = rot @ scale_matrix(1 / scale, (0, 0, 0))[:3, :3]
            # both kubric and mitsuba defaults to bounds with size 2
            # DSL uses default bound size 1 but that doesn't matter since we don't use DSL call parameters here
            kb_e = kb.Cube(
                name=f'shape_{eid:03d}', 
                position=translate, scale=scale, quaternion=pyquaternion.Quaternion(matrix=rot),
                static=eid == len(self.shape_scene) - 1)  # Assume the last shape is floor
            self.kb_scene.add(kb_e)
            # kb.move_until_no_overlap(kb_e, self.kb_engine, spawn_region=world_bounds)
            # shape_leaf = transform_shape(shape_leaf, translation_matrix(kb_e.position - compute_shape_center(shape_leaf)))

            assert np.allclose(shape_leaf[0]['to_world'], kb_e.matrix_world @ scale_matrix(kb_e.scale, (0, 0, 0)), atol=1e-3)

            self.shape_leaves.append(shape_leaf)
        self.shape_scene = concat_shapes(*self.shape_leaves)
        return self.shape_scene, {}
    
    def _simulate(self):
        _, info = self.kb_engine.run(frame_end=1)
        if len(info) > 0:
            print('collision detected!', len(info))
            # for item in info:
            #     print(item['instances'][0].asset_id, item['instances'][1].asset_id)

    def _synchronize(self):
        for eid in range(len(self.shape_leaves)):
            shape_leaf = self.shape_leaves[eid]
            kb_e = self.kb_scene.assets[eid]
            # shape_leaf = transform_shape(shape_leaf, translation_matrix(kb_e.position - compute_shape_center(shape_leaf)))
            # shape_leaf[0]['to_world'] = kb_e.matrix_world @ scale_matrix(kb_e.scale, (0, 0, 0))  # DO NOT mutate shapes themselves!!!!! It won't render properly after gathering frames
            shape_leaf = transform_shape(shape_leaf, kb_e.matrix_world @ scale_matrix(kb_e.scale, (0, 0, 0)) @ np.linalg.inv(shape_leaf[0]['to_world']))
            self.shape_leaves[eid] = shape_leaf
        self.shape_scene = concat_shapes(*self.shape_leaves)

    def step(self, action: str):
        # ignore action for now
        self._simulate()
        self._synchronize()
        return self.shape_scene, 0, False, False, {}

    def render(self):
        # can use `execute_from_preset` to render `self.shape_scene` into pixels
        pass


def main():
    save_dir = Path(PROJ_DIR) / 'logs/simulate_pybullet/test'
    save_dir.mkdir(parents=True, exist_ok=True)

    shapes = []

    with PyBulletEnv() as env:
        shape, _ = env.reset()
        shapes.append(shape)
        for _ in range(80):
            shape, *_ = env.step(action="")
            shapes.append(shape)

    # if using imports from `scripts.prompts.dsl_utils`, the animation function is not registered for `core` (which imports `dsl_utils` directly)
    sys.path.insert(0, (Path(PROJ_DIR) / 'scripts/prompts').as_posix())
    from dsl_utils import register_animation
    @register_animation()
    def history():
        return shapes
    core([], overwrite=True, save_dir=save_dir.as_posix())


if __name__ == '__main__':
    main()
