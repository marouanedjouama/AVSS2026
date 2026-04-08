"""
BraTS2020 dataset loader for preprocessed 2D slices.

Expects preprocessed data from preprocess_brats.py with structure:
    directory/
        BraTS20_Training_001_slice030.npz
        BraTS20_Training_001_slice031.npz
        ...
        BraTS20_Training_002_slice030.npz
        ...

Each .npz file contains:
    - image: (4, 256, 256) float32 - [flair, t1, t1ce, t2]
    - label: (3, 256, 256) uint8 - one-hot encoded [TC, WT, ET]
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms as monai_transforms
import nibabel as nib

class BraTSDataset2D(Dataset):
    def __init__(
        self,
        case_paths,
        train=True,
        slice_range=(0,154), 
        target_size=(256, 256),
    ):

        self.case_paths = case_paths

        if len(self.case_paths) == 0:
            raise ValueError(f"No case files found in {case_paths}")
    

        self.train = train
        self.slice_range = slice_range
        self.target_size = target_size

        self.num_slices = slice_range[1] - slice_range[0] + 1
        self.num_samples = len(self.case_paths) * self.num_slices 

        # Build transforms
        self._build_transforms()

    def _build_transforms(self):
        """Build transforms once during init."""
        # Spatial transforms (applied to both image and label)
        if self.train:
            self.transform = monai_transforms.Compose([
                monai_transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys =["label"]),
                monai_transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai_transforms.SpatialPadd(keys =["image", "label"], spatial_size=self.target_size, mode="constant"),
                monai_transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=self.target_size),

                monai_transforms.RandRotated(
                    keys=["image", "label"],
                    range_x=0.26,   # ±15 degrees in radians
                    prob=0.5,
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                    keep_size=True
                ),
                
                monai_transforms.RandFlipd(keys =["image", "label"], prob=0.5, spatial_axis=0),
                monai_transforms.RandFlipd(keys =["image", "label"], prob=0.5, spatial_axis=1),

                monai_transforms.RandZoomd(
                    keys=["image", "label"],
                    prob=0.3,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    mode=("bilinear", "nearest"),
                    keep_size=True
                ),

                monai_transforms.NormalizeIntensityd(keys =["image"], nonzero=True, channel_wise=True),

                monai_transforms.RandScaleIntensityd(keys =["image"], factors=0.1, prob=0.8),
                monai_transforms.RandShiftIntensityd(keys =["image"], offsets=0.1, prob=0.8),
                monai_transforms.RandGaussianNoised(keys=["image"], prob=0.5, std=0.15, sample_std=True),
                monai_transforms.RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.65, 1.5)),


                monai_transforms.ToTensord(keys=["image", "label"],),
            ])
        else:
            self.transform = monai_transforms.Compose([
                monai_transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys =["label"]),
                monai_transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai_transforms.SpatialPadd(keys =["image", "label"], spatial_size=self.target_size, mode="constant"),
                monai_transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=self.target_size),
                monai_transforms.NormalizeIntensityd(keys =["image"], nonzero=True, channel_wise=True),
                monai_transforms.ToTensord(keys=["image", "label"],),
            ])


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        case_idx = idx // self.num_slices
        slice_idx = idx % self.num_slices
        case_path = self.case_paths[case_idx]
        case_path = self.case_paths[case_idx]
        
        modalities_paths = [f.path for f in os.scandir(case_path)]
        modalities_paths_dict = {
            f.split("_")[-1].replace(".nii", ""): f for f in modalities_paths
        }

        # Load modalities
        t1 = nib.load(modalities_paths_dict["t1"]).get_fdata().astype(np.float32)
        t2 = nib.load(modalities_paths_dict["t2"]).get_fdata().astype(np.float32)
        t1ce = (
            nib.load(modalities_paths_dict["t1ce"]).get_fdata().astype(np.float32)
        )
        flair = nib.load(modalities_paths_dict["flair"]).get_fdata().astype(np.float32)
        seg = nib.load(modalities_paths_dict["seg"]).get_fdata().astype(np.int32)

        # Stack channels into shape (4, H, W, D)
        image = np.stack([flair, t1, t1ce, t2], axis=0)

        image = torch.from_numpy(image).permute(0, 3, 1, 2)#.float()  # (4, D, H, W)
        seg = (
            torch.from_numpy(seg).permute(2, 0, 1).unsqueeze(0)#.long()
        )  # (1, D, H, W)

        # extract slice_range slices
        image = image[:, self.slice_range[0] : self.slice_range[1] + 1, :, :]
        seg = seg[:, self.slice_range[0] : self.slice_range[1] + 1, :, :]

        # Select a slice along the depth dimension
        image = image[:, slice_idx, :, :]  # (4, H, W)
        seg = seg[0, slice_idx, :, :]  # (H, W) - remove channel dim for BraTS conversion

        data_input = {"image": image, "label": seg}
        data_input = self.transform(data_input)

        return data_input


if __name__ == "__main__":
    DATA_DIR = "/home/sbekhouche/Projects/marouane/BraTS2020_TrainingData"

    case_dirs = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    # Example: training dataset with augmentation
    train_dataset = BraTSDataset2D(
        case_paths=case_dirs,
        train=True,
        slice_range=(30, 135),  # Use all slices
    )

    print(f"Train samples: {len(train_dataset)}")
        
        
    idx = random.randint(0, len(train_dataset) - 1) 


    sample = train_dataset[idx]
    print(f"Image shape: {sample['image'].shape}")  # (4, 256, 256)
    print(f"Label shape: {sample['label'].shape}")  # (3, 256, 256)

    # Visualization
    import matplotlib.pyplot as plt

    image = sample['image'].numpy()  # (4, H, W)
    label = sample['label'].numpy()  # (3, H, W)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Image modalities
    modality_names = ['FLAIR', 'T1', 'T1ce', 'T2']
    for i, (ax, name) in enumerate(zip(axes[0], modality_names)):
        ax.imshow(image[i], cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    # Row 2: Label channels + combined overlay
    label_names = ['TC (Tumor Core)', 'WT (Whole Tumor)', 'ET (Enhancing)']
    colors = ['red', 'green', 'blue']
    for i, (ax, name) in enumerate(zip(axes[1, :3], label_names)):
        ax.imshow(label[i], cmap='hot')
        ax.set_title(name)
        ax.axis('off')

    # Combined RGB overlay on FLAIR
    axes[1, 3].imshow(image[0], cmap='gray')
    overlay = np.stack([label[0], label[1], label[2]], axis=-1).astype(float)
    axes[1, 3].imshow(overlay, alpha=0.5)
    axes[1, 3].set_title('Combined Overlay')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('brats_sample_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to brats_sample_visualization.png")
