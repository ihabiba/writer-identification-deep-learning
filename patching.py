from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Patch configuration
@dataclass(frozen=True)
class PatchConfig:
    patch_h: int = 128
    patch_w: int = 256

    seg_per_line: int = 8

    min_line_h: int = 10
    dilate_kernel: Tuple[int, int] = (40, 3)
    proj_threshold_frac: float = 0.02
    x_pad: int = 10

    min_ink_ratio: float = 0.01
    ink_gray_threshold: int = 200


# Binarization
def binarize_otsu_inv(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    _, bw = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return bw


# Ink ratio computation
def ink_ratio(gray: np.ndarray, ink_gray_threshold: int = 200) -> float:
    if gray.size == 0:
        return 0.0
    return float((gray < ink_gray_threshold).sum()) / float(gray.size)


# Text line detection
def find_text_lines(
    bw: np.ndarray, cfg: PatchConfig
) -> List[Tuple[int, int]]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, cfg.dilate_kernel
    )
    bw2 = cv2.dilate(bw, kernel, iterations=1)

    proj = np.sum(bw2 > 0, axis=1).astype(np.float32)
    maxv = float(proj.max()) if proj.size else 0.0
    thr = cfg.proj_threshold_frac * (maxv if maxv > 0 else 1.0)
    is_text = proj > thr

    lines: List[Tuple[int, int]] = []
    in_run = False
    y0 = 0
    h = bw2.shape[0]

    for y in range(h):
        if is_text[y] and not in_run:
            in_run = True
            y0 = y
        elif (not is_text[y]) and in_run:
            in_run = False
            y1 = y
            if (y1 - y0) >= cfg.min_line_h:
                lines.append((y0, y1))

    if in_run:
        y1 = h
        if (y1 - y0) >= cfg.min_line_h:
            lines.append((y0, y1))

    return lines


# Horizontal tight cropping
def tight_crop_x(
    bw_region: np.ndarray, x_pad: int = 10
) -> Optional[Tuple[int, int]]:
    cols = np.sum(bw_region > 0, axis=0)
    xs = np.where(cols > 0)[0]
    if xs.size == 0:
        return None

    x0 = max(0, int(xs.min()) - x_pad)
    x1 = min(bw_region.shape[1], int(xs.max()) + x_pad)
    if x1 <= x0:
        return None

    return x0, x1


# Height normalization without resizing
def crop_or_pad_to_height(
    gray_line: np.ndarray, target_h: int
) -> np.ndarray:
    h, w = gray_line.shape[:2]

    if h == target_h:
        return gray_line

    if h > target_h:
        y0 = (h - target_h) // 2
        return gray_line[y0:y0 + target_h, :]

    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    return np.pad(
        gray_line,
        ((pad_top, pad_bot), (0, 0)),
        constant_values=255,
    )


# Line to patch conversion
def line_to_patches_no_resize(
    line_gray: np.ndarray, cfg: PatchConfig
) -> List[np.ndarray]:
    line_fixed_h = crop_or_pad_to_height(
        line_gray, cfg.patch_h
    )
    h, w = line_fixed_h.shape

    if w < cfg.patch_w:
        pad = cfg.patch_w - w
        line_fixed_h = np.pad(
            line_fixed_h, ((0, 0), (0, pad)), constant_values=255
        )
        w = cfg.patch_w

    max_x0 = w - cfg.patch_w
    if max_x0 <= 0 or cfg.seg_per_line <= 1:
        xs = [0]
    else:
        stride = max(
            1, max_x0 // (cfg.seg_per_line - 1)
        )
        xs = list(range(0, max_x0 + 1, stride))[
            : cfg.seg_per_line
        ]

    patches: List[np.ndarray] = []
    for x0 in xs:
        p = line_fixed_h[:, x0:x0 + cfg.patch_w]
        if ink_ratio(p, cfg.ink_gray_threshold) >= cfg.min_ink_ratio:
            patches.append(p)

    return patches


# Full page patch extraction
def extract_line_patches_from_gray(
    gray: np.ndarray, cfg: PatchConfig
) -> List[np.ndarray]:
    bw = binarize_otsu_inv(gray)
    lines = find_text_lines(bw, cfg)

    patches: List[np.ndarray] = []
    H, W = gray.shape[:2]

    for (y0, y1) in lines:
        y0p = max(0, y0 - 4)
        y1p = min(H, y1 + 4)

        bw_line = bw[y0p:y1p, :]
        crop = tight_crop_x(bw_line, cfg.x_pad)
        if crop is None:
            continue

        x0, x1 = crop
        line_gray = gray[y0p:y1p, x0:x1]
        patches.extend(
            line_to_patches_no_resize(line_gray, cfg)
        )

    # Fallback patch extraction
    if len(patches) == 0:
        n_strips = 10
        step = max(1, H // n_strips)

        for i in range(n_strips):
            ys = i * step
            ye = H if i == n_strips - 1 else (i + 1) * step
            strip = gray[ys:ye, :]

            strip_fixed_h = crop_or_pad_to_height(
                strip, cfg.patch_h
            )
            patches.extend(
                line_to_patches_no_resize(strip_fixed_h, cfg)
            )

        if len(patches) == 0:
            full_fixed_h = crop_or_pad_to_height(
                gray, cfg.patch_h
            )
            patches.extend(
                line_to_patches_no_resize(full_fixed_h, cfg)
            )

    return patches
