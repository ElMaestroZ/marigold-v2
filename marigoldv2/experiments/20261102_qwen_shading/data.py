import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from marigoldv2.core.registry import register

_RGB_NAME_RE = re.compile(r"^rgb_(?P<camera>.+)_fr(?P<frame>\d+)$")


@register("manifest_transform")
class LoadHypersimShadingManifest:
    """Load RGB/shading manifests in the Marigold V1.1 IID format."""

    def __init__(
        self,
        base_dir,
        split,
        filelist_path=None,
        source_split=None,
        selection="all",
        holdout_samples=0,
        max_samples=None,
        seed=42,
        verify_files=True,
        allow_extra_columns=True,
    ):
        self.base_dir = Path(base_dir)
        self.split = str(split)
        self.source_split = str(source_split or split)
        self.filelist_path = (
            Path(filelist_path)
            if filelist_path is not None
            else self.base_dir / f"hypersim_filtered_{self.source_split}.txt"
        )
        if selection not in {"all", "exclude_holdout", "holdout"}:
            raise ValueError(f"Unknown manifest selection: {selection}")
        self.selection = selection
        self.holdout_samples = int(holdout_samples)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.seed = int(seed)
        self.verify_files = bool(verify_files)
        self.allow_extra_columns = bool(allow_extra_columns)

    def _resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.base_dir / path

    def _metadata(self, rgb_path, shading_path):
        relative_paths = []
        for label, path in (("RGB", rgb_path), ("shading", shading_path)):
            try:
                rel = path.relative_to(self.base_dir)
            except ValueError as exc:
                raise ValueError(
                    f"{label} path is outside shading root: {path}"
                ) from exc
            if len(rel.parts) != 3:
                raise ValueError(
                    f"Expected {label} path '<physical_split>/<scene>/<filename>', "
                    f"got '{rel.as_posix()}'"
                )
            relative_paths.append(rel)

        rgb_rel, shading_rel = relative_paths
        physical_split, scene, filename = rgb_rel.parts
        if rgb_rel.parts[:2] != shading_rel.parts[:2]:
            raise ValueError(
                "RGB/shading directory mismatch: "
                f"'{rgb_rel.as_posix()}' vs '{shading_rel.as_posix()}'"
            )
        allowed_splits = (
            {"train", "val"} if self.source_split == "train" else {self.source_split}
        )
        if physical_split not in allowed_splits:
            raise ValueError(
                f"Manifest source split mismatch: expected one of "
                f"{sorted(allowed_splits)}, found '{physical_split}' in "
                f"'{rgb_rel.as_posix()}'"
            )
        match = _RGB_NAME_RE.match(Path(filename).stem)
        if match is None:
            raise ValueError(f"Unexpected Hypersim RGB filename: {filename}")
        return physical_split, scene, match.group("camera"), match.group("frame")

    @staticmethod
    def _check_pair(rgb_path, shading_path):
        rgb_id = Path(rgb_path).stem.removeprefix("rgb_")
        shading_id = Path(shading_path).stem.removeprefix("shading_")
        if rgb_id != shading_id:
            raise ValueError(
                f"RGB/shading frame mismatch: '{rgb_path.name}' vs '{shading_path.name}'"
            )

    def __call__(self, manifest):
        if not self.filelist_path.is_file():
            raise FileNotFoundError(
                f"Shading manifest does not exist: {self.filelist_path}"
            )

        entries = []
        with self.filelist_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                columns = line.split()
                if len(columns) < 2 or (
                    len(columns) > 2 and not self.allow_extra_columns
                ):
                    raise ValueError(
                        f"{self.filelist_path}:{line_number}: expected at least two "
                        f"columns (RGB shading), got {len(columns)}"
                    )
                rgb_path, shading_path = map(self._resolve, columns[:2])
                stats_path = self._resolve(columns[2]) if len(columns) >= 3 else None
                self._check_pair(rgb_path, shading_path)
                if self.verify_files:
                    to_check = [rgb_path, shading_path]
                    if stats_path is not None:
                        to_check.append(stats_path)
                    missing = [str(p) for p in to_check if not p.is_file()]
                    if missing:
                        raise FileNotFoundError(
                            f"{self.filelist_path}:{line_number}: missing files: {missing}"
                        )
                physical_split, scene, camera, frame = self._metadata(
                    rgb_path, shading_path
                )
                entry = {
                    "split": self.split,
                    "source_split": self.source_split,
                    "physical_split": physical_split,
                    "scene": scene,
                    "camera": camera,
                    "frame": frame,
                    "rgb_path": str(rgb_path),
                    "shading_path": str(shading_path),
                }
                if stats_path is not None:
                    entry["stats_path"] = str(stats_path)
                entries.append(entry)

        if not entries:
            raise ValueError(f"No samples found in {self.filelist_path}")
        if self.holdout_samples < 0 or self.holdout_samples >= len(entries):
            if self.holdout_samples != 0:
                raise ValueError(
                    f"holdout_samples must be in [0, {len(entries) - 1}], "
                    f"got {self.holdout_samples}"
                )
        if self.selection != "all":
            indices = list(range(len(entries)))
            random.Random(self.seed).shuffle(indices)
            held_out = set(indices[: self.holdout_samples])
            keep_holdout = self.selection == "holdout"
            entries = [
                entry
                for index, entry in enumerate(entries)
                if (index in held_out) == keep_holdout
            ]
        if self.max_samples is not None and self.max_samples < len(entries):
            indices = list(range(len(entries)))
            random.Random(self.seed).shuffle(indices)
            selected = set(indices[: self.max_samples])
            entries = [
                entry for index, entry in enumerate(entries) if index in selected
            ]

        for index, entry in enumerate(entries):
            key = (
                f"hypersim_shading_{entry['split']}_{entry['scene']}_"
                f"{entry['camera']}_{entry['frame']}_{index:06d}"
            )
            if key in manifest:
                raise ValueError(f"Duplicate manifest key: {key}")
            manifest[key] = entry
        return manifest


def _clamp_shading_to_unit(shading: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Clamp finite linear shading to [0, 1] without shifting physical zero irradiance."""
    return torch.where(
        valid.expand_as(shading),
        shading.clamp(0.0, 1.0),
        torch.zeros_like(shading),
    )


@register("dataset_transform")
class ReadRGBShadingNpy:
    """Read linear RGB diffuse shading and preserve radiometric scale."""

    def __init__(
        self,
        key="shading_path",
        output_key="shading_gt",
        mask_key="valid_mask",
        normalize_to_unit=False,
    ):
        self.key = str(key)
        self.output_key = str(output_key)
        self.mask_key = str(mask_key)
        self.normalize_to_unit = bool(normalize_to_unit)

    def __call__(self, sample):
        path = sample["annotation"][self.key]
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected HWC shading with three channels in {path}, got {array.shape}"
            )
        shading = torch.from_numpy(np.asarray(array, dtype=np.float32)).permute(2, 0, 1)
        valid = torch.isfinite(shading).all(dim=0, keepdim=True)
        
        # Clean non-finite elements
        clean = torch.where(
            valid.expand_as(shading),
            torch.nan_to_num(shading, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(shading),
        )
        
        # Strictly clamp without subtracting min (preserving physical zero)
        clean = _clamp_shading_to_unit(clean, valid)
        sample[self.output_key] = clean
        sample[self.mask_key] = valid
        return sample


@register("dataset_transform")
class ReadGrayscaleShadingNpy:
    """Read linear RGB shading, convert to Rec.709 gray, and repeat as RGB."""

    def __init__(
        self,
        key="shading_path",
        output_key="shading_gt",
        mask_key="valid_mask",
        normalize_to_unit=False,
    ):
        self.key = str(key)
        self.output_key = str(output_key)
        self.mask_key = str(mask_key)
        self.normalize_to_unit = bool(normalize_to_unit)

    def __call__(self, sample):
        path = sample["annotation"][self.key]
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected HWC shading with three channels in {path}, got {array.shape}"
            )
        shading = torch.from_numpy(np.asarray(array, dtype=np.float32)).permute(2, 0, 1)
        valid = torch.isfinite(shading).all(dim=0, keepdim=True)
        safe = torch.where(torch.isfinite(shading), shading, torch.zeros_like(shading))
        gray = 0.2126 * safe[0:1] + 0.7152 * safe[1:2] + 0.0722 * safe[2:3]
        gray = _clamp_shading_to_unit(gray, valid)
        sample[self.output_key] = gray.repeat(3, 1, 1)
        sample[self.mask_key] = valid
        return sample


@register("dataset_transform")
class RandomHorizontalFlipShading:
    """Flip aligned image, shading, and mask tensors without channel transforms."""

    def __init__(self, p=0.5, include_keys=None, exclude_keys=None):
        self.p = float(p)
        self.include_keys = None if include_keys is None else set(include_keys)
        self.exclude_keys = set(exclude_keys or [])

    def __call__(self, sample):
        do_flip = random.random() < self.p
        sample["hflip_applied"] = do_flip
        if not do_flip:
            return sample
        for key, value in list(sample.items()):
            if not torch.is_tensor(value) or value.ndim < 2:
                continue
            if "orig" in key or key in self.exclude_keys:
                continue
            if self.include_keys is not None and key not in self.include_keys:
                continue
            sample[key] = torch.flip(value, dims=[-1])
        return sample


@register("dataset_transform")
class ResizeRGBShading:
    """Resize RGB/shading bilinearly and validity masks with nearest neighbor."""

    def __init__(
        self,
        height=576,
        width=768,
        rgb_keys=("rgb_norm", "rgb_int"),
        shading_key="shading_gt",
        mask_key="valid_mask",
    ):
        self.height = int(height)
        self.width = int(width)
        self.rgb_keys = list(rgb_keys)
        self.shading_key = str(shading_key)
        self.mask_key = str(mask_key)

    @staticmethod
    def _resize(value, size, mode):
        kwargs = {"size": size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(value.unsqueeze(0).float(), **kwargs).squeeze(0)

    def __call__(self, sample):
        size = (self.height, self.width)
        for key in self.rgb_keys:
            value = sample.get(key)
            if value is None:
                continue
            resized = self._resize(value, size, "bilinear")
            sample[key] = (
                resized.to(value.dtype)
                if torch.is_floating_point(value)
                else resized.round().clamp(0, 255).to(value.dtype)
            )
        shading = sample.get(self.shading_key)
        mask = sample.get(self.mask_key)
        if shading is None or mask is None:
            raise KeyError(
                f"ResizeRGBShading requires '{self.shading_key}' and '{self.mask_key}'"
            )
        if shading.ndim != 3 or shading.shape[0] != 3:
            raise ValueError(f"Expected shading [3,H,W], got {tuple(shading.shape)}")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"Expected mask [1,H,W], got {tuple(mask.shape)}")
        
        resized_shading = self._resize(shading, size, "bilinear").clamp(0.0, 1.0)
        resized_mask = self._resize(mask.float(), size, "nearest") > 0.5
        sample[self.shading_key] = torch.where(
            resized_mask.expand_as(resized_shading),
            resized_shading,
            torch.zeros_like(resized_shading),
        ).to(shading.dtype)
        sample[self.mask_key] = resized_mask
        return sample