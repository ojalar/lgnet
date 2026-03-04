import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transforms = transforms
        
        self.images = sorted(os.listdir(image_dir))
        self.masks  = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        sample = {
            "image": image,
            "mask": mask
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        image = sample["image"]
        mask  = sample["mask"]

        # ensure mask is integer class IDs
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = (mask > 0).long()

        return {"image": image, "mask": mask, "image_id": self.images[idx]}

def make_collate_fn(processor):
    def collate_fn(batch):
        images = [x["image"] for x in batch]
        masks  = [x["mask"]  for x in batch]
        ids = [x["image_id"] for x in batch]

        encoded = processor(images=images, segmentation_maps=masks, return_tensors="pt")
        return encoded, images, masks, ids

    return collate_fn

def make_collate_fn_test(processor):
    def collate_fn(batch):
        images = [x["image"] for x in batch]
        masks  = [x["mask"]  for x in batch]
        ids = [x["image_id"] for x in batch]

        encoded = processor(images=images, return_tensors="pt")
        return encoded, images, masks, ids

    return collate_fn