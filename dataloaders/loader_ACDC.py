"""
Fast ACDC dataset loader for preprocessed 2D slices.
Use after running preprocess_ACDC.py.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms
import matplotlib.pyplot as plt


PREPROCESSED_DIR = "data/ACDC/database/ACDC_preprocessed"


def get_train_augmentations():
    """Runtime augmentations for training (random transforms only)."""
    return transforms.Compose(
        [
            # transforms.RandRotated(
            #     keys=["image", "label"],
            #     range_x=0.26,   # ±15 degrees in radians
            #     prob=0.5,
            #     mode=("bilinear", "nearest"),
            #     padding_mode="zeros",
            #     keep_size=True
            # ),
            transforms.RandFlipd(keys =["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys =["image", "label"], prob=0.5, spatial_axis=1),
            # transforms.RandZoomd(
            #     keys=["image", "label"],
            #     prob=0.3,
            #     min_zoom=0.9,
            #     max_zoom=1.1,
            #     mode=("bilinear", "nearest"),
            #     keep_size=True
            # ),

            transforms.RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(0.26,),   # ~15degrees
                scale_range=(0.1,),
                translate_range=(10, 10),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",

            ),
            transforms.Rand2DElasticd(
                keys=["image", "label"],
                prob=0.3,
                spacing=(20, 20),
                magnitude_range=(1, 3),
                mode=("bilinear", "nearest"),
                padding_mode="zeros"
            ),

            transforms.RandScaleIntensityd(keys =["image"], factors=0.1, prob=0.5), # 0.8 -> 0.5
            transforms.RandShiftIntensityd(keys =["image"], offsets=0.1, prob=0.5), # 0.8 -> 0.5
            transforms.RandGaussianNoised(keys=["image"], prob=0.3, std=0.08, sample_std=True), # prob 0.5 -> 0.3, std 0.10 -> 0.8
            transforms.RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.4)), # increased prob from 0.2 to 0.3| (0.65, 1.5) ->(0.7, 1.4)
            
            transforms.EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )


def get_val_augmentations():
    """No augmentations for validation - just type conversion."""
    return transforms.Compose(
        [
            transforms.EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )


class ACDCPreprocessed(Dataset):
    """
    Fast dataset loader for preprocessed ACDC slices.

    Args:
        data_dir: Path to preprocessed directory (e.g., ACDC_preprocessed/training)
        transform: Optional MONAI transform (use get_train_augmentations() or get_val_augmentations())
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Find all .npz files
        self.samples = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        )

        if len(self.samples) == 0:
            raise ValueError(
                f"No .npz files found in {data_dir}. "
                "Did you run preprocess_ACDC.py first?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load preprocessed slice
        file_path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(file_path)

        image = data["image"]  # (H, W) float32
        label = data["label"]  # (H, W) uint8

        # Add channel dimension: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        sample = {
            "image": image,
            "label": label.astype(np.float32),  # MONAI transforms expect float
        }

        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["label"]


def visualize_sample(image, label, output_dir, idx=0):
    """Visualize a sample and save to disk."""
    os.makedirs(output_dir, exist_ok=True)

    img_np = image.squeeze().numpy()
    lbl_np = label.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Image
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Label (classes: 0=BG, 1=RV, 2=Myo, 3=LV)
    axes[1].imshow(lbl_np, cmap="tab10", vmin=0, vmax=3)
    axes[1].set_title("Label (RV=1, Myo=2, LV=3)")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img_np, cmap="gray")
    masked = np.ma.masked_where(lbl_np == 0, lbl_np)
    axes[2].imshow(masked, cmap="tab10", alpha=0.5, vmin=0, vmax=3)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sample_{idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    # Example usage
    train_dir = os.path.join(PREPROCESSED_DIR, "train")
    val_dir = os.path.join(PREPROCESSED_DIR, "test")

    # Training dataset with augmentations
    train_dataset = ACDCPreprocessed(
        data_dir=train_dir,
        transform=get_train_augmentations(),
    )

    # Validation dataset without augmentations
    val_dataset = ACDCPreprocessed(
        data_dir=val_dir,
        transform=get_val_augmentations(),
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Test loading
    idx = random.randint(0, len(train_dataset) - 1)
    image, label = train_dataset[idx]
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label dtype: {label.dtype}")
    print(f"Label unique values: {torch.unique(label).tolist()}")

    # Visualize
    samples_dir = os.path.join(os.path.dirname(__file__), "ACDC_samples_fast")
    visualize_sample(image, label, samples_dir, idx)
