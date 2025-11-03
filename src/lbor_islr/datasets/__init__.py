from typing import Dict, Any, Tuple

from torch.utils.data import DataLoader

from .wlasl import WLASLDataset

__all__ = [
    "WLASLDataset",
    "build_dataloaders_from_config",
]


def build_dataloaders_from_config(
    cfg: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
   
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})

    name = str(dataset_cfg.get("name", "wlasl")).lower()
    root = dataset_cfg.get("root", "data/wlasl")
    train_list = dataset_cfg.get("train_list", "")
    val_list = dataset_cfg.get("val_list", "")
    num_frames = int(dataset_cfg.get("num_frames", 52))

    if name != "wlasl":
        raise ValueError(
            f"build_dataloaders_from_config currently only supports name='wlasl', "
            f"got '{name}'. Please extend datasets/__init__.py for other datasets."
        )

    if not train_list or not val_list:
        raise ValueError(
            "Please set 'dataset.train_list' and 'dataset.val_list' in the config."
        )

    train_dataset = WLASLDataset(
        root=root,
        split_file=train_list,
        num_frames=num_frames,
        normalize=True,
    )
    val_dataset = WLASLDataset(
        root=root,
        split_file=val_list,
        num_frames=num_frames,
        normalize=True,
    )

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
