import time
import os
import json
import pickle
from typing import Tuple

import numpy as np
import torch

import detectron2.data.transforms as T
from detectron2.data.detection_utils import read_image
from torch.utils.data import Dataset, DataLoader


class EpicKitchen(Dataset):
    """ Dataset for epic kitchen
    """
    LENGTH_FRAME_ID = 10
    CLASS_NAME = 'Person'

    def __init__(self, cfg, part: str, clip: str, basedir='/mnt/seagate12t/EPIC-KITCHEN/EPIC-KITCHEN') -> None:
        super(EpicKitchen, self).__init__()
        info = json.load(open(os.path.join(basedir, 'info.json'), 'r'))
        self.cfg = cfg
        self.basedir = os.path.join(basedir, info[part][clip]['path'])
        self.total_count = info[part][clip]['count']
        self._init_transform()

    def _init_transform(self):
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        case = read_image(os.path.join(self.basedir, self._index_to_img_string(1)), format="BGR")
        self.transform = self.aug.get_transform(case)
        self.img_height, self.img_width = case.shape[:2]

    def _index_to_img_string(self, index) -> str:
        return f'frame_{index + 1:0{self.LENGTH_FRAME_ID}d}.jpg'

    def __len__(self):
        return self.total_count

    def __getitem__(self, index) -> Tuple:
        img_path = os.path.join(self.basedir, self._index_to_img_string(index))
        st = time.time()
        img = read_image(img_path, format="BGR")
        # ic('0', time.time() - st)
        img = self.transform.apply_image(img)
        # ic('1', time.time() - st)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        # ic('2', time.time() - st)
        return {"image": img, "height": self.img_height, "width": self.img_width, "class_names": self.CLASS_NAME}
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


