"""
dataset.py
==========
NTIRE 2026 데이터셋 관련 공통 모듈
모든 노트북에서 아래처럼 import해서 사용:

    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders()
"""

import os
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ── ImageNet 정규화 상수 ────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transform 정의 ──────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class AIGenDetDataset(Dataset):
    """
    NTIRE 2026 Train Dataset (shard 구조)
    label: 0 = Real, 1 = AI Generated
    """
    def __init__(self, shard_dir: str, shard_nums=None, transform=None):
        self.shard_root = shard_dir
        self.transform  = transform

        if shard_nums is None:
            shard_dirs = [os.path.join(shard_dir, f"shard_{i}") for i in range(6)]
        else:
            shard_dirs = [os.path.join(shard_dir, f"shard_{i}") for i in shard_nums]

        # 실제 존재하는 shard만 사용
        shard_dirs = [x for x in shard_dirs if os.path.isdir(x)]

        dfs = []
        for sp in shard_dirs:
            df = pd.read_csv(os.path.join(sp, "labels.csv"), index_col=0)
            df["shard_name"] = Path(sp).name
            dfs.append(df)

        self.label_df = pd.concat(dfs, ignore_index=True)
        print(f"[Dataset] shard {len(shard_dirs)}개 로드 완료 — 총 {len(self.label_df):,}장")

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row      = self.label_df.iloc[idx]
        img_path = os.path.join(
            self.shard_root, row["shard_name"], "images", row["image_name"]
        )
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    shard_root:  str  = "../data/train",
    shard_nums        = [0],
    batch_size:  int  = 32,
    val_ratio:   float = 0.1,
    test_ratio:  float = 0.1,
    seed:        int  = 42,
    num_workers: int  = 0,       # Windows 환경: 0 권장
):
    """
    DataLoader 한 번에 반환하는 헬퍼 함수.

    Returns:
        train_loader, val_loader, test_loader,
        train_ds, val_ds, test_ds
    """
    full_dataset = AIGenDetDataset(
        shard_dir  = shard_root,
        shard_nums = shard_nums,
        transform  = train_transform
    )

    total      = len(full_dataset)
    train_size = int((1 - val_ratio - test_ratio) * total)
    val_size   = int(val_ratio * total)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Val / Test 는 증강 없이 정규화만
    val_ds.dataset.transform  = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
