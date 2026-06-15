"""Dataset preparation and loading helpers."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, LABEL_MAP

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PIECE_PATTERN = re.compile(r"^([A-Z\-]+\d+)")
THAI_FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{4})_(?P<class_code>FR|HF|SP)_(?P<source_code>PK|UP)_(?P<piece_id>P\d+)\.[A-Za-z0-9]+$"
)


def iter_image_paths(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def extract_kaggle_class(filename: str) -> str:
    name = Path(filename).stem.upper()
    if name.startswith("HALF-FRESH") or name.startswith("HALFFRESH"):
        return "HALF_FRESH"
    if name.startswith("FRESH"):
        return "FRESH"
    if name.startswith("SPOILED"):
        return "SPOILED"
    return "UNKNOWN"


def extract_piece_id(filename: str) -> str:
    stem = Path(filename).stem.upper()
    match = PIECE_PATTERN.match(stem)
    return match.group(1) if match else stem


def prepare_kaggle_dataframe(data_dir: str | Path) -> pd.DataFrame:
    records = []
    for path in iter_image_paths(data_dir):
        class_name = extract_kaggle_class(path.name)
        if class_name == "UNKNOWN":
            continue
        records.append(
            {
                "filename": path.name,
                "path": str(path.resolve()),
                "class": class_name,
                "piece_id": extract_piece_id(path.name),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No usable images found under {data_dir}")
    return df.reset_index(drop=True)


def group_split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["piece_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    overlap = set(train_df["piece_id"]) & set(val_df["piece_id"])
    if overlap:
        raise ValueError(f"Leakage detected across splits: {sorted(overlap)[:5]}")
    return train_df, val_df


def build_train_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ClassificationDataset(Dataset):
    """Standard image classification dataset backed by a CSV dataframe."""

    def __init__(self, df: pd.DataFrame, transform: T.Compose | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = LABEL_MAP[row["class"]]
        return image, label


class PredictionDataset(Dataset):
    """Dataset for unlabeled prediction on files or folders."""

    def __init__(self, image_paths: list[Path], transform: T.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path.resolve())


@dataclass
class CvatImageRecord:
    filename: str
    width: int
    height: int
    polygons: list[list[tuple[float, float]]]


def parse_cvat_xml(xml_path: str | Path) -> list[CvatImageRecord]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records: list[CvatImageRecord] = []
    for image_el in root.findall("image"):
        polygons = []
        for polygon_el in image_el.findall("polygon"):
            points = polygon_el.get("points", "")
            polygon = [tuple(map(float, pt.split(","))) for pt in points.split(";") if pt]
            polygons.append(polygon)
        records.append(
            CvatImageRecord(
                filename=os.path.basename(image_el.get("name", "")),
                width=int(image_el.get("width")),
                height=int(image_el.get("height")),
                polygons=polygons,
            )
        )
    return records


def polygons_to_mask(polygons: list[list[tuple[float, float]]], height: int, width: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        drawer.polygon(polygon, fill=255, outline=255)
    return np.asarray(mask, dtype=np.uint8)


def parse_thai_filename(filename: str) -> dict[str, str]:
    match = THAI_FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(
            "Expected Thai filename format YYYYMMDD_HHMM_classCode_sourceType_pieceID.ext"
        )
    class_code = match.group("class_code")
    source_code = match.group("source_code")
    class_map = {"FR": "Fresh", "HF": "Half-Fresh", "SP": "Spoiled"}
    source_map = {"PK": "Packaged", "UP": "Unpackaged"}
    return {
        "class_code": class_code,
        "class": class_map[class_code],
        "source_code": source_code,
        "source": source_map[source_code],
        "piece_id": match.group("piece_id"),
    }


def prepare_phase2_manifest(
    thai_data_dir: str | Path,
    cvat_xml_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, Path]:
    thai_data_dir = Path(thai_data_dir)
    output_dir = Path(output_dir)
    gt_masks_dir = output_dir / "gt_masks"
    gt_masks_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_cvat_xml(cvat_xml_path)
    records = []
    for item in parsed:
        if not item.polygons:
            continue
        image_path = thai_data_dir / item.filename
        if not image_path.exists():
            continue
        try:
            metadata = parse_thai_filename(item.filename)
        except ValueError:
            continue
        mask = polygons_to_mask(item.polygons, item.height, item.width)
        mask_path = gt_masks_dir / f"{Path(item.filename).stem}_mask.png"
        Image.fromarray(mask).save(mask_path)
        records.append(
            {
                "filename": item.filename,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "width": item.width,
                "height": item.height,
                "n_polygons": len(item.polygons),
                "class": metadata["class"],
                "class_code": metadata["class_code"],
                "source": metadata["source"],
                "source_code": metadata["source_code"],
                "piece_id": metadata["piece_id"],
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid CVAT/image pairs found; check XML filenames and image folder")
    return df, gt_masks_dir


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"path", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    invalid = set(df["class"].unique()) - set(CLASS_NAMES)
    if invalid:
        raise ValueError(f"{csv_path} contains unsupported classes: {sorted(invalid)}")
    return df.reset_index(drop=True)
