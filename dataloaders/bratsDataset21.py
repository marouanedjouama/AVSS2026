"""
BraTS2021 dataset loader for preprocessed 2D slices.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms as monai_transforms
import nibabel as nib

class BraTSDataset21(Dataset):
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

        modalities_paths = [f.path for f in os.scandir(case_path)]
        modalities_paths_dict = {
            f.split("-")[-1].replace(".nii.gz", ""): f for f in modalities_paths
        }

        # Load modalities
        t1c = nib.load(modalities_paths_dict["t1c"]).get_fdata().astype(np.float32)
        t1n = nib.load(modalities_paths_dict["t1n"]).get_fdata().astype(np.float32)
        t2f = (
            nib.load(modalities_paths_dict["t2f"]).get_fdata().astype(np.float32)
        )
        t2w = nib.load(modalities_paths_dict["t2w"]).get_fdata().astype(np.float32)
        seg = nib.load(modalities_paths_dict["seg"]).get_fdata().astype(np.int32)

        # Stack channels into shape (4, H, W, D)
        image = np.stack([t1c, t1n, t2f, t2w], axis=0)

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

        # Convert seg to one-hot encoding with 3 classes (TC, WT, ET)
        seg_one_hot = torch.zeros((3, seg.shape[0], seg.shape[1]), dtype=torch.float32)
        seg_one_hot[0] = ((seg == 1) | (seg == 3)).float()  # TC
        seg_one_hot[1] = ((seg == 1) | (seg == 2) | (seg == 3)).float()  # WT
        seg_one_hot[2] = (seg == 3).float()  # ET

        # or
        # npimage = np.load(img_path)
        # npmask = np.load(mask_path)
        # npimage = npimage.transpose((2, 0, 1))

        # WT_Label = npmask.copy()
        # WT_Label[npmask == 1] = 1.
        # WT_Label[npmask == 2] = 1.
        # WT_Label[npmask == 4] = 1.
        # TC_Label = npmask.copy()
        # TC_Label[npmask == 1] = 1.
        # TC_Label[npmask == 2] = 0.
        # TC_Label[npmask == 4] = 1.
        # ET_Label = npmask.copy()
        # ET_Label[npmask == 1] = 0.
        # ET_Label[npmask == 2] = 0.
        # ET_Label[npmask == 4] = 1.
        # nplabel = np.empty((224, 224, 3))
        # nplabel[:, :, 0] = WT_Label
        # nplabel[:, :, 1] = TC_Label
        # nplabel[:, :, 2] = ET_Label
        # nplabel = nplabel.transpose((2, 0, 1))

        data_input = {"image": image, "label": seg_one_hot}
        data_input = self.transform(data_input)

        return data_input


if __name__ == "__main__":
    data_paths = ["data/brats2021/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000", "data/brats2021/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000"]
    dataset = BraTSDataset21(data_paths, train=True, slice_range=(30,130), target_size=(192,192))
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
    print(f"Label shape: {sample['label'].shape}, dtype: {sample['label'].dtype}")
