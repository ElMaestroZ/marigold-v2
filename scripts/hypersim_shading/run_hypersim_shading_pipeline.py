#!/usr/bin/env python3
"""Build the Hypersim RGB/shading dataset from the Marigold V1.1 IID split.

Downloads only ``diffuse_illumination`` per scene, processes it into float32
shading targets (plus optional JSON stats), and hard-links RGB from the
preprocessed depth dataset. Scenes are skipped on rerun once complete.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SHADING_MODALITY = ".diffuse_illumination.hdf5"
MIN_FREE_BYTES = 500_000_000_000  # ~135 GB staging + ~280 GB float32 targets
SAMPLE_RE = re.compile(
    r"^(?P<split>train|val|test)/(?P<scene>ai_\d{3}_\d{3})/"
    r"rgb_(?P<camera>cam_\d{2})_fr(?P<frame>\d{4})\.png$"
)
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Sample:
    split: str
    scene: str
    camera: str
    frame: int
    rgb_path: str
    shading_path: str
    stats_path: str


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selective-downloader", type=Path, required=True)
    parser.add_argument("--filtered-list-dir", type=Path, required=True)
    parser.add_argument("--rgb-dataset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="float32 is lossless for Hypersim's float16 source",
    )
    parser.add_argument(
        "--normalize-percentile",
        type=float,
        default=None,
        help=(
            "If set (e.g. 99.0), divide shading by this percentile and clip to [0, 1]. "
            "If unset, keep nonnegative linear HDR values (>= 0.0)."
        ),
    )
    parser.add_argument(
        "--write-stats",
        action="store_true",
        help="Also write shading_stats_<cam>_fr<frame>.json alongside shading .npy files",
    )
    return parser.parse_args()


def parse_sample_line(path: Path, line_number: int, line: str) -> Sample:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(
            f"{path}:{line_number}: expected five IID paths, got {len(fields)}"
        )
    rgb_path, albedo_path, shading_path, residual_path, stats_path = fields
    match = SAMPLE_RE.fullmatch(rgb_path)
    if match is None:
        raise ValueError(f"{path}:{line_number}: malformed RGB path: {rgb_path}")

    metadata = match.groupdict()
    frame = int(metadata["frame"])
    expected_stem = f"{metadata['camera']}_fr{frame:04d}"
    expected = (
        f"{metadata['split']}/{metadata['scene']}/albedo_{expected_stem}.npy",
        f"{metadata['split']}/{metadata['scene']}/shading_{expected_stem}.npy",
        f"{metadata['split']}/{metadata['scene']}/residual_{expected_stem}.npy",
        f"{metadata['split']}/{metadata['scene']}/shading_stats_{expected_stem}.json",
    )
    actual = (albedo_path, shading_path, residual_path, stats_path)
    if actual != expected:
        raise ValueError(
            f"{path}:{line_number}: modalities do not identify the same sample; "
            f"expected {expected}, got {actual}"
        )
    return Sample(
        split=metadata["split"],
        scene=metadata["scene"],
        camera=metadata["camera"],
        frame=frame,
        rgb_path=rgb_path,
        shading_path=shading_path,
        stats_path=stats_path,
    )


def read_list(path: Path) -> list[Sample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    samples = [
        parse_sample_line(path, line_number, line)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.strip()
    ]
    if not samples:
        raise ValueError(f"{path}: empty sample list")
    if len(samples) != len({sample.rgb_path for sample in samples}):
        raise ValueError(f"{path}: duplicate RGB entries")
    return samples


def load_curated_lists(
    filtered_list_dir: Path,
) -> tuple[dict[str, list[Sample]], dict[str, list[Sample]]]:
    named = {
        "train": read_list(filtered_list_dir / "hypersim_train_filtered.txt"),
        "test": read_list(filtered_list_dir / "hypersim_test.txt"),
        "val": read_list(filtered_list_dir / "hypersim_val.txt"),
        "vis": read_list(filtered_list_dir / "hypersim_vis.txt"),
    }
    canonical = named["train"] + named["test"]
    by_rgb = {sample.rgb_path: sample for sample in canonical}
    if len(by_rgb) != len(canonical):
        raise ValueError("train and test lists overlap")
    for list_name in ("val", "vis"):
        unknown = [
            sample.rgb_path
            for sample in named[list_name]
            if sample.rgb_path not in by_rgb
        ]
        if unknown:
            raise ValueError(
                f"{list_name}: {len(unknown)} samples are absent from train/test"
            )

    by_scene: dict[str, list[Sample]] = defaultdict(list)
    for sample in canonical:
        by_scene[sample.scene].append(sample)
    return named, dict(by_scene)


def check_free_space(output_dir: Path) -> None:
    free = shutil.disk_usage(output_dir.parent).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Need {MIN_FREE_BYTES / 1e9:.0f} GB free under {output_dir.parent}; "
            f"{free / 1e9:.0f} GB available"
        )


def raw_shading_path(raw_dir: Path, sample: Sample) -> Path:
    return (
        raw_dir
        / sample.scene
        / "images"
        / f"scene_{sample.camera}_final_hdf5"
        / f"frame.{sample.frame:04d}.diffuse_illumination.hdf5"
    )


def validate_nonempty(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(
            f"{label}: {len(missing)} missing/empty files; first paths:\n{preview}"
        )


def run_with_retries(command: list[str], log_path: Path, retries: int) -> None:
    for attempt in range(1, retries + 1):
        with log_path.open("a") as handle:
            handle.write(f"\nAttempt {attempt}: {' '.join(command)}\n")
            handle.flush()
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode == 0:
            return
        if attempt < retries:
            delay = min(60, 5 * (2 ** (attempt - 1)))
            log(f"[retry] command failed ({result.returncode}); retrying in {delay}s")
            time.sleep(delay)
    raise subprocess.CalledProcessError(result.returncode, command)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def atomic_write_json(destination: Path, payload: dict[str, float]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def save_shading(
    source: Path,
    destination: Path,
    stats_destination: Path | None,
    dtype: np.dtype,
    split: str,
    normalize_percentile: float | None,
) -> None:
    with h5py.File(source, "r") as handle:
        if "dataset" not in handle:
            raise ValueError(f"{source}: missing HDF5 dataset")
        shading = np.asarray(handle["dataset"])
    if shading.ndim != 3 or shading.shape[-1] != 3:
        raise ValueError(f"{source}: expected HWC RGB shading, got {shading.shape}")

    shading = np.maximum(shading.astype(dtype, copy=False), 0.0)
    if split in {"train", "val"} and not np.isfinite(shading).all():
        raise ValueError(f"{source}: non-finite value in curated training shading")

    finite_vals = shading[np.isfinite(shading)]
    if finite_vals.size == 0:
        raise ValueError(f"{source}: shading contains no finite values")

    stats = {
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "mean": float(np.mean(finite_vals)),
        "p90": float(np.percentile(finite_vals, 90.0)),
        "p99": float(np.percentile(finite_vals, 99.0)),
        "p999": float(np.percentile(finite_vals, 99.9)),
    }

    if normalize_percentile is not None:
        scale = float(np.percentile(finite_vals, normalize_percentile))
        stats["norm_percentile"] = float(normalize_percentile)
        stats["norm_scale"] = scale
        if scale > 1e-6:
            shading = np.clip(shading / scale, 0.0, 1.0).astype(dtype, copy=False)
        else:
            shading = np.clip(shading, 0.0, 1.0).astype(dtype, copy=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, shading, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)

    if stats_destination is not None:
        atomic_write_json(stats_destination, stats)


def expected_sample_outputs(
    output_dir: Path, sample: Sample, write_stats: bool
) -> tuple[Path, ...]:
    if write_stats:
        return (
            output_dir / sample.rgb_path,
            output_dir / sample.shading_path,
            output_dir / sample.stats_path,
        )
    return (output_dir / sample.rgb_path, output_dir / sample.shading_path)


def validate_scene_outputs(
    output_dir: Path, samples: list[Sample], scene: str, write_stats: bool
) -> None:
    paths = [
        path
        for sample in samples
        for path in expected_sample_outputs(output_dir, sample, write_stats)
    ]
    validate_nonempty(paths, f"{scene} outputs")


def process_scene(
    scene: str,
    samples: list[Sample],
    args: argparse.Namespace,
) -> tuple[str, str]:
    raw_dir = args.work_dir / "raw"
    state_dir = args.work_dir / "state"
    scene_stage = args.output_dir / ".staging" / scene
    done_path = state_dir / "done" / f"{scene}.done"
    raw_ready_path = state_dir / "raw_ready" / f"{scene}.ready"
    scene_log = state_dir / "logs" / f"{scene}.log"

    if done_path.is_file():
        validate_scene_outputs(args.output_dir, samples, scene, args.write_stats)
        return scene, "already complete"

    done_path.parent.mkdir(parents=True, exist_ok=True)
    raw_ready_path.parent.mkdir(parents=True, exist_ok=True)
    scene_log.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_raw = [raw_shading_path(raw_dir, sample) for sample in samples]

    if raw_ready_path.is_file():
        validate_nonempty(expected_raw, f"{scene} reusable raw shading")
    else:
        command = [
            sys.executable,
            str(args.selective_downloader),
            "--directory",
            str(raw_dir),
            "--scene",
            scene,
            "--contains",
            SHADING_MODALITY,
            "--overwrite",
            "--silent",
        ]
        run_with_retries(command, scene_log, args.download_retries)
        validate_nonempty(expected_raw, f"{scene} raw shading")
        raw_ready_path.write_text(
            f"scene={scene}\nrequired_files={len(expected_raw)}\n"
        )

    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    scene_stage.mkdir(parents=True)
    dtype = np.dtype(args.output_dtype)
    for sample in samples:
        source_rgb = args.rgb_dataset_dir / sample.rgb_path
        if not source_rgb.is_file() or source_rgb.stat().st_size == 0:
            raise FileNotFoundError(f"missing preprocessed RGB: {source_rgb}")
        staged_rgb = scene_stage / Path(sample.rgb_path).name
        staged_shading = scene_stage / Path(sample.shading_path).name
        staged_stats = (
            scene_stage / Path(sample.stats_path).name if args.write_stats else None
        )
        link_or_copy(source_rgb, staged_rgb)
        save_shading(
            raw_shading_path(raw_dir, sample),
            staged_shading,
            staged_stats,
            dtype,
            sample.split,
            args.normalize_percentile,
        )

    for split in {sample.split for sample in samples}:
        split_samples = [sample for sample in samples if sample.split == split]
        target_scene = args.output_dir / split / scene
        target_scene.parent.mkdir(parents=True, exist_ok=True)
        split_stage = scene_stage / split
        split_stage.mkdir()
        for sample in split_samples:
            names = [Path(sample.rgb_path).name, Path(sample.shading_path).name]
            if args.write_stats:
                names.append(Path(sample.stats_path).name)
            for name in names:
                os.replace(scene_stage / name, split_stage / name)
        if target_scene.exists():
            shutil.rmtree(target_scene)
        os.replace(split_stage, target_scene)

    validate_scene_outputs(args.output_dir, samples, scene, args.write_stats)
    done_path.write_text(f"scene={scene}\nsamples={len(samples)}\n")

    raw_scene = raw_dir / scene
    if raw_scene.exists():
        shutil.rmtree(raw_scene)
    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    return scene, f"processed {len(samples)} samples"


def format_manifest_line(sample: Sample, write_stats: bool) -> str:
    if write_stats:
        return f"{sample.rgb_path} {sample.shading_path} {sample.stats_path}"
    return f"{sample.rgb_path} {sample.shading_path}"


def write_manifests(
    output_dir: Path, named: dict[str, list[Sample]], write_stats: bool
) -> None:
    output_names = {
        "train": "hypersim_filtered_train.txt",
        "test": "hypersim_filtered_test.txt",
        "val": "hypersim_filtered_val.txt",
        "vis": "hypersim_filtered_vis.txt",
    }
    for list_name, filename in output_names.items():
        lines = [
            format_manifest_line(sample, write_stats) for sample in named[list_name]
        ]
        (output_dir / filename).write_text("\n".join(lines) + "\n")
    canonical = named["train"] + named["test"]
    (output_dir / "hypersim_filtered_all.txt").write_text(
        "\n".join(format_manifest_line(sample, write_stats) for sample in canonical)
        + "\n"
    )


def validate_dataset(
    output_dir: Path, named: dict[str, list[Sample]], write_stats: bool
) -> None:
    canonical = named["train"] + named["test"]
    paths = [
        path
        for sample in canonical
        for path in expected_sample_outputs(output_dir, sample, write_stats)
    ]
    validate_nonempty(paths, "final RGB/shading dataset")
    if len(named["train"]) != 23_842 or len(named["test"]) != 5_238:
        raise RuntimeError(
            "upstream IID split count changed: "
            f"train={len(named['train'])}, test={len(named['test'])}"
        )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in (
        args.selective_downloader,
        args.filtered_list_dir,
        args.rgb_dataset_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".staging").mkdir(exist_ok=True)
    check_free_space(args.output_dir)
    named, by_scene = load_curated_lists(args.filtered_list_dir)
    log(
        f"Starting {len(by_scene)} scenes with {args.workers} workers; "
        f"{len(named['train'])} train, {len(named['test'])} test, "
        f"{len(named['val'])} validation-subset samples"
    )

    failures: list[tuple[str, BaseException]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_scene, scene, samples, args): scene
            for scene, samples in sorted(by_scene.items())
        }
        for future in as_completed(futures):
            scene = futures[future]
            try:
                _, status = future.result()
                completed += 1
                log(f"[{completed}/{len(futures)}] {scene}: {status}")
            except BaseException as error:
                failures.append((scene, error))
                log(f"[FAILED] {scene}: {error!r}")

    if failures:
        for scene, error in failures:
            log(f"Failure summary: {scene}: {error!r}")
        raise RuntimeError(
            f"{len(failures)} scenes failed; rerun to retry incomplete scenes"
        )

    write_manifests(args.output_dir, named, args.write_stats)
    validate_dataset(args.output_dir, named, args.write_stats)
    shutil.rmtree(args.output_dir / ".staging")
    (args.output_dir / "PREPROCESS_COMPLETE").write_text(
        f"scenes={len(by_scene)}\ntrain_samples={len(named['train'])}\n"
        f"test_samples={len(named['test'])}\nval_subset_samples={len(named['val'])}\n"
        f"dtype={args.output_dtype}\n"
        f"normalize_percentile={args.normalize_percentile}\n"
        f"write_stats={args.write_stats}\n"
    )
    log("All RGB/shading pairs validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())