import glob
import random
import os
from functools import reduce
from itertools import chain
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.io import read_image


class ChestXRayImageDataset(VisionDataset):
    rel_img_dir = 'static'

    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
              'Pneumothorax', 'none']

    def __init__(
        self,
        root: str,
        data_frame: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(ChestXRayImageDataset,
              self).__init__(root, transform=transform,
                             target_transform=target_transform)

        self.img_dir = os.path.join(root, self.rel_img_dir)

        for label in self.labels:
            data_frame[label] = data_frame['findings'].map(lambda finding: 1.0 if label in finding else 0.0)

        self.data = data_frame


    def _load_data(self, frac: float = 1.) -> Tuple[Any, Any]:
        return data.iloc[:, 0], data.iloc[:, 2:17]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = os.path.join(self.img_dir, self.data.iloc[index, 0])
        img_path = glob.glob(img_path)
        img = Image.open(img_path[0]).convert('RGB')

        target = torch.tensor(self.data.iloc[index, 2:17].values.astype(np.float32))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target