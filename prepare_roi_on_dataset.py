from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ROI LED dataset for YOLO training.")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(r"D:\Desktop2\Images\roi_crops"),
        help="Directory that contains ROI crop images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(r"d:\led-visual-inspection\roi_labeled\labels"),
        help="Directory that contains YOLO txt labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"d:\led-visual-inspection\dataset_roi_on"),
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        target_dir = root / relative
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)


def collect_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in supported_suffixes:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        pairs.append((image_path, label_path))

    return pairs


def write_data_yaml(root: Path) -> None:
    data_yaml = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: 'on'\n"
    )
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")


def copy_subset(pairs: list[tuple[Path, Path]], root: Path, subset: str) -> None:
    image_target_dir = root / "images" / subset
    label_target_dir = root / "labels" / subset

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_target_dir / image_path.name)
        shutil.copy2(label_path, label_target_dir / label_path.name)


def main() -> None:
    args = parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image directory not found: {args.images}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label directory not found: {args.labels}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    pairs = collect_pairs(args.images, args.labels)
    if not pairs:
        raise RuntimeError("No matched image/label pairs were found.")

    random.Random(args.seed).shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_ratio))
    train_pairs = pairs[:-val_count]
    val_pairs = pairs[-val_count:]

    ensure_dirs(args.output)
    write_data_yaml(args.output)
    copy_subset(train_pairs, args.output, "train")
    copy_subset(val_pairs, args.output, "val")

    print(f"Matched pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Dataset root: {args.output}")


if __name__ == "__main__":
    main()
