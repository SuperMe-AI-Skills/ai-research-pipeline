"""
data_prep.py
============
Data preparation utilities for image classification tasks.

Handles dataset creation, transforms, folder loading, train/val/test splitting,
DataLoader construction, and basic image statistics.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ===================================================================== #
#  Dataset                                                               #
# ===================================================================== #
class ImageDataset(Dataset):
    """Custom ``Dataset`` that loads images from file paths and labels.

    Parameters
    ----------
    image_paths : Sequence[str]
        Absolute or relative paths to image files.
    labels : Sequence[int]
        Integer labels corresponding to each image.
    transform : Optional[transforms.Compose]
        Torchvision transform pipeline applied to every image.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        labels: Sequence[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        assert len(image_paths) == len(labels), (
            f"image_paths ({len(image_paths)}) and labels ({len(labels)}) "
            "must have the same length."
        )
        self.image_paths: List[str] = list(image_paths)
        self.labels: List[int] = list(labels)
        self.transform = transform

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ===================================================================== #
#  Transforms                                                            #
# ===================================================================== #
def build_transforms(
    target_size: int = 224,
    mode: str = "train",
) -> transforms.Compose:
    """Build a torchvision transform pipeline.

    Parameters
    ----------
    target_size : int
        Spatial resolution for the output tensor (height == width).
    mode : str
        One of ``'train'``, ``'val'``, or ``'test'``.
        * *train* applies random augmentations (flip, rotation, jitter).
        * *val* / *test* apply deterministic resize + centre-crop.

    Returns
    -------
    transforms.Compose
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((target_size + 32, target_size + 32)),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


# ===================================================================== #
#  Folder loading                                                        #
# ===================================================================== #
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


def load_image_folder(
    root_dir: str,
) -> Tuple[List[str], List[int], List[str]]:
    """Load an ImageFolder-style directory tree.

    Expected layout::

        root_dir/
            class_a/
                img001.jpg
                img002.jpg
            class_b/
                img003.jpg
                ...

    Parameters
    ----------
    root_dir : str
        Root directory containing one sub-folder per class.

    Returns
    -------
    image_paths : List[str]
        Sorted list of absolute image file paths.
    labels : List[int]
        Corresponding integer label for each image.
    class_names : List[str]
        Sorted list of class names (sub-folder names).
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    class_names = sorted(
        [d.name for d in root.iterdir() if d.is_dir()]
    )
    if not class_names:
        raise ValueError(f"No class sub-directories found in {root_dir}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths: List[str] = []
    labels: List[int] = []

    for cls_name in class_names:
        cls_dir = root / cls_name
        for fpath in sorted(cls_dir.iterdir()):
            if fpath.suffix.lower() in _IMAGE_EXTENSIONS and fpath.is_file():
                image_paths.append(str(fpath))
                labels.append(class_to_idx[cls_name])

    if not image_paths:
        raise ValueError(f"No images found in {root_dir}")

    return image_paths, labels, class_names


# ===================================================================== #
#  Train / Val / Test split                                              #
# ===================================================================== #
def split_image_data(
    image_paths: Sequence[str],
    labels: Sequence[int],
    test_size: float = 0.20,
    val_size_of_train: float = 0.25,
    random_state: int = 2025,
    stratify: bool = True,
) -> Tuple[
    List[str], List[str], List[str],
    List[int], List[int], List[int],
]:
    """Split image paths and labels into train / val / test sets.

    The split is performed in two stages:

    1. Separate *test_size* fraction as the test set.
    2. From the remaining data, separate *val_size_of_train* fraction as the
       validation set.

    With the defaults (``test_size=0.20``, ``val_size_of_train=0.25``) the
    effective proportions are **60 / 20 / 20**.

    Parameters
    ----------
    image_paths, labels : array-like
        Parallel sequences of paths and integer labels.
    test_size : float
        Fraction of the full data reserved for testing.
    val_size_of_train : float
        Fraction of the *remaining* data (after test split) reserved for
        validation.
    random_state : int
        Seed for reproducibility.
    stratify : bool
        Whether to stratify splits by label distribution.

    Returns
    -------
    train_paths, val_paths, test_paths,
    train_labels, val_labels, test_labels
    """
    strat = labels if stratify else None

    train_val_paths, test_paths, train_val_labels, test_labels = (
        train_test_split(
            image_paths,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    )

    strat_tv = train_val_labels if stratify else None
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size_of_train,
        random_state=random_state,
        stratify=strat_tv,
    )

    return (
        list(train_paths),
        list(val_paths),
        list(test_paths),
        list(train_labels),
        list(val_labels),
        list(test_labels),
    )


# ===================================================================== #
#  DataLoader factory                                                    #
# ===================================================================== #
def create_dataloaders(
    train_paths: Sequence[str],
    train_labels: Sequence[int],
    val_paths: Sequence[str],
    val_labels: Sequence[int],
    test_paths: Sequence[str],
    test_labels: Sequence[int],
    target_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test ``DataLoader`` instances.

    Training data is shuffled; validation and test data are not.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_tf = build_transforms(target_size=target_size, mode="train")
    eval_tf = build_transforms(target_size=target_size, mode="val")

    train_ds = ImageDataset(train_paths, train_labels, transform=train_tf)
    val_ds = ImageDataset(val_paths, val_labels, transform=eval_tf)
    test_ds = ImageDataset(test_paths, test_labels, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ===================================================================== #
#  Image statistics                                                      #
# ===================================================================== #
def get_image_stats(
    image_paths: Sequence[str],
    n_sample: int = 100,
) -> Dict[str, Any]:
    """Compute basic channel statistics and size distribution.

    A random subset of *n_sample* images is used to estimate per-channel
    means and standard deviations (in [0, 1] space) as well as the
    distribution of image dimensions.

    Parameters
    ----------
    image_paths : Sequence[str]
        All image paths available.
    n_sample : int
        Number of images to sample for statistics.

    Returns
    -------
    dict
        ``channel_means``, ``channel_stds`` (each length-3 lists),
        ``widths`` and ``heights`` (lists of sampled image dimensions),
        ``n_sampled`` (int).
    """
    paths = list(image_paths)
    if len(paths) > n_sample:
        paths = random.sample(paths, n_sample)

    pixel_sums = np.zeros(3, dtype=np.float64)
    pixel_sq_sums = np.zeros(3, dtype=np.float64)
    pixel_count: int = 0
    widths: List[int] = []
    heights: List[int] = []

    to_tensor = transforms.ToTensor()  # scales to [0, 1]

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        widths.append(w)
        heights.append(h)

        tensor = to_tensor(img)  # [3, H, W]
        npx = tensor.shape[1] * tensor.shape[2]
        pixel_count += npx

        for c in range(3):
            chan = tensor[c].numpy().astype(np.float64)
            pixel_sums[c] += chan.sum()
            pixel_sq_sums[c] += (chan ** 2).sum()

    if pixel_count == 0:
        return {
            "channel_means": [0.0, 0.0, 0.0],
            "channel_stds": [1.0, 1.0, 1.0],
            "widths": [],
            "heights": [],
            "n_sampled": 0,
        }

    means = pixel_sums / pixel_count
    stds = np.sqrt(pixel_sq_sums / pixel_count - means ** 2)

    return {
        "channel_means": means.tolist(),
        "channel_stds": stds.tolist(),
        "widths": widths,
        "heights": heights,
        "n_sampled": len(widths),
    }
