from __future__ import annotations

from collections.abc import Iterator

import numpy as np


VIEW_SPECS = (
    ("Sagittal", 2),
    ("Coronal", 1),
    ("Axial", 0),
)


def normalize_volume_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Match PigBET preprocessing normalization for slice export."""
    data = np.asarray(volume)
    data_min = data.min()
    data_max = data.max()
    if data_max == data_min:
        return np.zeros_like(data, dtype=np.uint8)
    return ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)


def extract_view_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 2:
        return volume[:, :, index]
    if axis == 1:
        return volume[:, index, :]
    if axis == 0:
        return volume[index, :, :]
    raise ValueError(f"Unsupported axis {axis}.")


def extract_rgb_slice_from_normalized(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    current = extract_view_slice(volume, axis, index)
    empty = np.zeros_like(current, dtype=np.uint8)

    previous = empty if index == 0 else extract_view_slice(volume, axis, index - 1)
    following = empty if index == volume.shape[axis] - 1 else extract_view_slice(volume, axis, index + 1)

    return np.stack([previous, current, following], axis=2)


def iter_rgb_slices_from_normalized(volume: np.ndarray, axis: int) -> Iterator[tuple[int, np.ndarray]]:
    for index in range(volume.shape[axis]):
        yield index, extract_rgb_slice_from_normalized(volume, axis, index)


def extract_middle_rgb_slice(volume: np.ndarray, axis: int) -> np.ndarray:
    normalized = normalize_volume_to_uint8(volume)
    middle_index = volume.shape[axis] // 2
    return extract_rgb_slice_from_normalized(normalized, axis, middle_index)


def build_middle_view_triptych(volume: np.ndarray) -> list[tuple[str, np.ndarray]]:
    normalized = normalize_volume_to_uint8(volume)
    return [
        (label, extract_rgb_slice_from_normalized(normalized, axis, normalized.shape[axis] // 2))
        for label, axis in VIEW_SPECS
    ]
