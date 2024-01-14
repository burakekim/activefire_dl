"""ActiveFire dataset."""

import glob
import os
import random
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset

MAX_PIXEL_VALUE = 65535


class ActiveFire(Dataset):
    """Dataset instance for the Active Fire dataset."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        split="train",
    ):
        """Initialize ActiveFire dataset instance introduced in "Active fire detection in Landsat-8 imagery: A large-scale dataset and a deep-learning study".

        Args:
            img_dir (str): Path to the images
            mask_dir (str): Path to the masks
            transforms (Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]], optional): Transformations. Defaults to None.
            split (str, optional): Split "train" "val" or "test". Defaults to "train".

        """  # noqa: E501
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.img_names = sorted(glob.glob(img_dir + "/*_RT_*.tif"))
        self.mask_names = sorted(glob.glob(mask_dir + "/*_RT_*.tif"))

        self.ids = []

        for mask_name in self.mask_names:
            self.ids.append(os.path.split(mask_name)[-1].split(".")[0].split("_")[-1])

        random.shuffle(self.ids)

        total_elements = len(self.ids)
        train_idx = int(0.7 * total_elements)
        val_idx = int(0.9 * total_elements)

        if split == "train":
            self.ids = self.ids[:train_idx]
            print("Train", len(self.ids))
        if split == "val":
            self.ids = self.ids[train_idx:val_idx]
            print("Validation", len(self.ids))
        if split == "test":
            self.ids = self.ids[val_idx:]
            print("Test", len(self.ids))

    def __len__(self):
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset

        """
        return len(self.ids)

    def __getitem__(self, idx):
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index

        """
        fname = self.ids[idx]

        im_id = [x for x in self.img_names if x.endswith(fname + ".tif")]
        mask_id = [x for x in self.mask_names if x.endswith(fname + ".tif")]

        img_path = os.path.join(self.img_dir, im_id[0])
        mask_path = os.path.join(self.mask_dir, mask_id[0])

        image = self.get_img_arr(img_path)
        mask = self.get_mask_arr(mask_path)

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample["image"], sample["mask"], self.ids[idx]

    def get_img_arr(self, path: str) -> Tensor:
        """Load a single raster or target.

        Args:
            path (str): the directory of the image or target

        Returns:
            image

        """
        img = rasterio.open(path).read()
        img = np.float32(img) / MAX_PIXEL_VALUE
        img = torch.from_numpy(img).float()
        return img

    def get_mask_arr(self, path: str) -> Tensor:
        """Load a single raster or target.

        Args:
            path (str): the directory of the image or target

        Returns
            image

        """
        img = rasterio.open(path).read()
        seg = np.float32(img)
        seg = torch.from_numpy(seg).float()
        return seg
