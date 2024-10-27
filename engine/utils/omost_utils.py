import os
from engine.third_party.omost.lib_omost.canvas import Canvas, valid_colors, safe_str
from typing import Union, Tuple
import shutil
from pathlib import Path
try:
    import cv2
except:
    print('[ERROR] failed to load cv2')
import numpy as np
import numpy.typing
from engine.constants import PROJ_DIR


class MyCanvas(Canvas):
    def set_global_description(self, description: str, detailed_descriptions: list[str], tags: str,
                               HTML_web_color_name: str = 'white'):
        super().set_global_description(description, detailed_descriptions, tags, HTML_web_color_name)

    def add_my_local_description(self,
                                 segm: np.typing.NDArray[bool],
                                 depth: np.typing.NDArray[np.float32],
                                 box: 'BBox',
                                 description: str,
                                 detailed_descriptions: list[str], tags: str, atmosphere: str, style: str,
                                 quality_meta: str, color: Union[Tuple[float, float, float], None],
                                 erode_kernel: int = 8):
        h, w = segm.shape
        rect = (box.min[1] / h, box.max[1] / h, box.min[0] / w, box.max[0] / w)
        rect = [max(0, min(90, i * 90)) for i in rect]
        mask = cv2.resize((segm * 255).astype(np.uint8), (90, 90), interpolation=cv2.INTER_AREA) > 127.5
        if mask.sum() < 9:
            print(f'[ERROR] mask sum < 4, skip: {description}')
            return

        if erode_kernel > 0:
            kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
            erode_segm = cv2.erode((segm * 255).astype(np.uint8), kernel) > 127.5
        else:
            erode_segm = segm
        distance_to_viewer = depth[erode_segm].min()  # FIXME this is probably wrong
        print(f'[INFO] distance_to_viewer for {description}: {distance_to_viewer}')

        color = np.array([[valid_colors[np.random.choice(list(valid_colors.keys()))] if color is None else (np.asarray(color) * 255)]], dtype=np.uint8)
        prefixes = self.prefixes + [description]
        suffixes = detailed_descriptions

        if self.record_tags:
            suffixes = suffixes + [tags, atmosphere, style, quality_meta]

        prefixes = [safe_str(x) for x in prefixes]
        suffixes = [safe_str(x) for x in suffixes]

        self.components.append(dict(
            mask=mask,
            rect=rect,
            distance_to_viewer=distance_to_viewer,
            color=color,
            prefixes=prefixes,
            suffixes=suffixes
        ))

    def dense_process(self):
        self.components = sorted(self.components, key=lambda x: x['distance_to_viewer'], reverse=True)

        # compute initial latent
        initial_latent = np.zeros(shape=(90, 90, 3), dtype=np.float32) + self.color

        for component in self.components:
            mask = component['mask']
            initial_latent[mask] = 0.7 * component['color'] + 0.3 * initial_latent[mask]

        initial_latent = initial_latent.clip(0, 255).astype(np.uint8)

        # compute conditions

        bag_of_conditions = [
            dict(mask=np.ones(shape=(90, 90), dtype=np.float32), prefixes=self.prefixes, suffixes=self.suffixes)
        ]

        for i, component in enumerate(self.components):
            bag_of_conditions.append(dict(
                mask=component['mask'],
                prefixes=component['prefixes'],
                suffixes=component['suffixes']
            ))
            print(f'[INFO] component {i:02d}')
            print(component['prefixes'], component['suffixes'])

        return dict(
            initial_latent=initial_latent,
            bag_of_conditions=bag_of_conditions,
        )
