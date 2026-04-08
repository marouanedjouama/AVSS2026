"""ISIC 2018 Dataset for skin lesion segmentation."""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from monai import transforms as monai_transforms
import torchvision.transforms.functional as TF


class ISIC2018Dataset(Dataset):
    """ISIC 2018 Challenge Dataset for skin lesion segmentation.

    Expects the following directory structure:
        data_path/
            ISIC2018_Task1-2_Training_Input/
                ISIC_0000000.jpg
                ISIC_0000001.jpg
                ...
            ISIC2018_Task1_Training_GroundTruth/
                ISIC_0000000_segmentation.png
                ISIC_0000001_segmentation.png
                ...

    Args:
        data_path: Path to the ISIC 2018 dataset root directory.
        target_size: Target size for resizing images (H, W).
        transform: Optional additional transforms.
        augment: Whether to apply data augmentation during training.
    """

    def __init__(
        self,
        data_path: str,
        target_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        augment: bool = True,
        split: str = "train",
    ):
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.transform = transform
        self.augment = augment
        self.split = split

        # Find image and mask directories

        if self.split == "train":
            self.image_dir = self.data_path / "ISIC2018_Task1-2_Training_Input"
            self.mask_dir = self.data_path / "ISIC2018_Task1_Training_GroundTruth"
        
        else:
            self.image_dir = self.data_path / "ISIC2018_Task1-2_Validation_Input"
            self.mask_dir = self.data_path / "ISIC2018_Task1_Validation_GroundTruth"

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Get list of image files
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        # Verify corresponding masks exist
        self.valid_samples = []
        for img_path in self.image_files:
            # ISIC naming convention: ISIC_0000000.jpg -> ISIC_0000000_segmentation.png
            mask_name = img_path.stem + "_segmentation.png"
            mask_path = self.mask_dir / mask_name
            if mask_path.exists():
                self.valid_samples.append((img_path, mask_path))

        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid image-mask pairs found in {data_path}")

        print(f"Found {len(self.valid_samples)} valid image-mask pairs")

        # Normalization for RGB images (ImageNet stats)
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )

        self.normalize = monai_transforms.ScaleIntensity()  # Scale to [0, 1]

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _apply_augmentation(self, image: Image.Image, mask: Image.Image):
        """Apply synchronized augmentation to image and mask."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (0, 90, 180, 270 degrees)
        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle > 0:
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random color jitter (only for image)
        if torch.rand(1) > 0.5:
            image = TF.adjust_brightness(image, 0.8 + torch.rand(1).item() * 0.4)
        if torch.rand(1) > 0.5:
            image = TF.adjust_contrast(image, 0.8 + torch.rand(1).item() * 0.4)
        if torch.rand(1) > 0.5:
            image = TF.adjust_saturation(image, 0.8 + torch.rand(1).item() * 0.4)

        return image, mask

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.valid_samples[idx]

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale

        # Resize to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Convert to tensors
        image = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        mask = TF.to_tensor(mask)    # [1, H, W] in [0, 1]

        # Normalize image
        image = self.normalize(image)

        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).float()

        # Apply additional transforms if provided
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": mask,
        }


class ISIC2018DatasetSimple(Dataset):
    """Simplified ISIC 2018 Dataset that returns tensors directly.

    This version doesn't use normalization to keep values in [0, 1] range,
    which is often better for diffusion models.
    """

    def __init__(
        self,
        data_path: str,
        target_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
    ):
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.augment = augment

        # Find image and mask directories
        self.image_dir = self.data_path / "ISIC2018_Task1-2_Training_Input"
        self.mask_dir = self.data_path / "ISIC2018_Task1_Training_GroundTruth"

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Get list of image files
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        # Verify corresponding masks exist
        self.valid_samples = []
        for img_path in self.image_files:
            mask_name = img_path.stem + "_segmentation.png"
            mask_path = self.mask_dir / mask_name
            if mask_path.exists():
                self.valid_samples.append((img_path, mask_path))

        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid image-mask pairs found in {data_path}")

        print(f"Found {len(self.valid_samples)} valid image-mask pairs")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _apply_augmentation(self, image: Image.Image, mask: Image.Image):
        """Apply synchronized augmentation to image and mask."""
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle > 0:
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.valid_samples[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Convert to tensors, scale to [-1, 1] for image
        image = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        image = image * 2 - 1  # Scale to [-1, 1]

        mask = TF.to_tensor(mask)  # [1, H, W] in [0, 1]
        mask = (mask > 0.5).float()

        return {
            "image": (image, None, None),
            "label": mask,
        }



if __name__ == "__main__":
    # Example usage of the ISIC2018Dataset
    data_path = "/home/sbekhouche/Projects/marouane/MIUA2026/datasets/ISIC_data"
    dataset = ISIC2018Dataset(data_path, target_size=(256, 256), augment=True, split="train")

    print(f"Dataset size: {len(dataset)}")

    import random

    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    image, label = sample["image"][0], sample["label"]
    print(f"Sample image shape: {image.shape}, label shape: {label.shape}")

    print("Sample image tensor range:", image.min().item(), image.max().item())
    print("Sample label unique values:", torch.unique(label))