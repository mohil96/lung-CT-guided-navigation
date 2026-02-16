"""Dataset and preprocessing utilities for 3D lung CT segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _resolve_case_pairs(split_dir: Path, mask_suffix: str) -> List[Tuple[Path, Path]]:
    ct_files = sorted(split_dir.glob("*_ct.nii.gz"))
    pairs: List[Tuple[Path, Path]] = []

    for ct_file in ct_files:
        mask_file = Path(str(ct_file).replace("_ct.nii.gz", mask_suffix))
        if mask_file.exists():
            pairs.append((ct_file, mask_file))

    if not pairs:
        raise ValueError(
            f"No paired CT/mask files found in {split_dir} with mask suffix '{mask_suffix}'"
        )

    return pairs


def load_nifti(filepath: Path) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
    nii = nib.load(str(filepath))
    volume = nii.get_fdata().astype(np.float32)
    spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    affine = nii.affine.astype(np.float32)
    return volume, spacing, affine


def resample_volume(
    volume: np.ndarray,
    original_spacing: Sequence[float],
    target_spacing: Sequence[float],
    is_mask: bool = False,
) -> np.ndarray:
    factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    order = 0 if is_mask else 1
    return zoom(volume, zoom=factors, order=order)


def window_and_normalize(volume: np.ndarray, intensity_range: Sequence[float]) -> np.ndarray:
    min_hu, max_hu = float(intensity_range[0]), float(intensity_range[1])
    clipped = np.clip(volume, min_hu, max_hu)
    return ((clipped - min_hu) / (max_hu - min_hu)).astype(np.float32)


def pad_to_shape(volume: np.ndarray, target_shape: Sequence[int], pad_value: float = 0.0) -> np.ndarray:
    pads = []
    for curr, target in zip(volume.shape, target_shape):
        total = max(target - curr, 0)
        before = total // 2
        after = total - before
        pads.append((before, after))
    return np.pad(volume, pads, mode="constant", constant_values=pad_value)


def random_crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    image = pad_to_shape(image, patch_size, pad_value=0.0)
    mask = pad_to_shape(mask, patch_size, pad_value=0.0)

    starts = []
    for curr, patch in zip(image.shape, patch_size):
        max_start = max(curr - patch, 0)
        starts.append(int(rng.integers(0, max_start + 1)) if max_start > 0 else 0)

    d, h, w = starts
    pd, ph, pw = patch_size
    return (
        image[d : d + pd, h : h + ph, w : w + pw],
        mask[d : d + pd, h : h + ph, w : w + pw],
    )


def center_crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = pad_to_shape(image, patch_size, pad_value=0.0)
    mask = pad_to_shape(mask, patch_size, pad_value=0.0)

    starts = []
    for curr, patch in zip(image.shape, patch_size):
        starts.append(max((curr - patch) // 2, 0))

    d, h, w = starts
    pd, ph, pw = patch_size
    return (
        image[d : d + pd, h : h + ph, w : w + pw],
        mask[d : d + pd, h : h + ph, w : w + pw],
    )


class SegmentationDataset(Dataset):
    """Generic dataset for CT + segmentation mask training."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        mask_suffix: str,
        patch_size: Sequence[int],
        target_spacing: Sequence[float],
        intensity_range: Sequence[float],
        seed: int = 42,
    ) -> None:
        self.split_dir = Path(data_dir) / split
        self.split = split
        self.mask_suffix = mask_suffix
        self.patch_size = tuple(int(x) for x in patch_size)
        self.target_spacing = tuple(float(x) for x in target_spacing)
        self.intensity_range = intensity_range
        self.rng = np.random.default_rng(seed)

        self.pairs = _resolve_case_pairs(self.split_dir, mask_suffix)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ct_path, mask_path = self.pairs[idx]

        ct, spacing, _ = load_nifti(ct_path)
        mask, mask_spacing, _ = load_nifti(mask_path)

        if spacing != mask_spacing:
            raise ValueError(
                f"Spacing mismatch for case {ct_path.name}: CT={spacing}, mask={mask_spacing}"
            )

        ct = resample_volume(ct, spacing, self.target_spacing, is_mask=False)
        mask = resample_volume(mask, spacing, self.target_spacing, is_mask=True)

        ct = window_and_normalize(ct, self.intensity_range)
        mask = (mask > 0.5).astype(np.float32)

        if self.split == "train":
            ct_patch, mask_patch = random_crop_pair(ct, mask, self.patch_size, self.rng)
        else:
            ct_patch, mask_patch = center_crop_pair(ct, mask, self.patch_size)

        ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)
        return ct_tensor, mask_tensor


def dataset_summary(dataset: SegmentationDataset) -> Dict[str, object]:
    return {
        "split_dir": str(dataset.split_dir),
        "split": dataset.split,
        "cases": len(dataset),
        "patch_size": dataset.patch_size,
        "target_spacing": dataset.target_spacing,
        "mask_suffix": dataset.mask_suffix,
    }
