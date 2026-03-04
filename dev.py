#!/usr/bin/env python

# Copyright 2026 [Risto Ojala / GitHub @ojalar, Tristan Ellison / GitHub @ tristan-ze]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import logging
import math
import os
import sys
import pickle
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from PIL import Image

from transformers import (
    AutoImageProcessor,
    get_scheduler,
)

import torch.nn as nn
from typing import List, Dict
import importlib.util
import sys
import os
import time
from matplotlib import pyplot as plt


import torch.nn.functional as F
from timm.optim import optim_factory
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import v2 as T

from models.LGNet import create_mask2former_dinov3_model, create_swin_mask2former, create_lgnet

# Import utility modules for metrics, visualization, and data loading
from utils.utils import CalibCurve, Metrics, m2foutput_to_prob_masks, Visualizer, compute_pixels_in_dataset
from utils.dataloader import SegmentationDataset, make_collate_fn, make_collate_fn_test

# Define label mappings for binary glass segmentation task (background vs glass)
ID2LABEL = {
    0: "background",
    1: "glass"
}

LABEL2ID = {
    "background" : 0,
    "glass" : 1
}

# Initialize the Mask2Former image processor from pre-trained weights
m2f_pretrained = "facebook/mask2former-swin-small-ade-semantic"
processor = AutoImageProcessor.from_pretrained(m2f_pretrained)

# Define standard image resolution for preprocessing (512x512)
H, W = (512, 512)
processor.size = {"height": H, "width": W}

# Define data augmentation pipeline for training (random flips)
train_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1)
])

def train(dataset_path: str, model_suffix: str, model_type: str, dinov3_size: str = "large", se: bool = True):
    """
    Train a glass segmentation model.
    
    Args:
        dataset_path: Path to dataset root (should contain train/image and train/mask subdirectories)
        model_suffix: Suffix for the saved model filename
        model_type: Type of model to train - "dinov3", "swin", or "lgnet"
        dinov3_size: Size of DINOv3 backbone if applicable - "large", "base", or "small"
        se: Whether to use Squeeze-Excitation blocks in L+GNet (only relevant for lgnet type)
    """

    # Map dinov3_size to a model name string (functions detect 'vitl', 'vitb', 'vits')
    default_dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    if dinov3_size == "large":
        dinov3_model_name = default_dinov3_name
    elif dinov3_size == "base":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vitb")
    elif dinov3_size == "small":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vits")
    else:
        raise ValueError(f"Unsupported dinov3_size: {dinov3_size}")

    # Select model constructor based on argument
    if model_type == "dinov3":
        model = create_mask2former_dinov3_model(ID2LABEL, LABEL2ID, dinov3_model_name)
    elif model_type == "swin":
        model = create_swin_mask2former(ID2LABEL, LABEL2ID)
    else:
        model = create_lgnet(ID2LABEL, LABEL2ID, dinov3_model_name, se=se)

    # Initialize training dataset with image/mask pairs and augmentation transforms
    train_dataset = SegmentationDataset(
        image_dir= os.path.join(dataset_path, "train/image"),
        mask_dir= os.path.join(dataset_path, "train/mask"),
        processor=processor,
        transforms=train_transforms
    )

    # Create data loader for batch processing during training
    collate_fn = make_collate_fn(processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Calculate total training steps for scheduler
    num_epochs = 30
    num_training_steps = num_epochs * len(train_dataloader)
    
    # Set up optimizer with weight decay applied to relevant parameters
    param_groups = optim_factory.param_groups_weight_decay(
        model,
        weight_decay=1e-4
    )

    # Initialize AdamW optimizer and linear learning rate scheduler with warmup
    optimizer = AdamW(param_groups, lr=1e-4)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )
    
    # Prepare model for training: enable gradients, move to GPU, initialize gradient scaler for mixed precision
    model.train()
    model.to("cuda")
    scaler = GradScaler()  # for gradient scaling

    # Training loop
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for batch, _, _, _ in train_dataloader:
            optimizer.zero_grad()
            # Move batch data to GPU
            batch["pixel_values"] = batch["pixel_values"].to("cuda")
            batch["pixel_mask"] = batch["pixel_mask"].to("cuda")
            batch["mask_labels"] = [mask_label.to("cuda") for mask_label in batch["mask_labels"]]
            batch["class_labels"] = [class_label.to("cuda") for class_label in batch["class_labels"]]
            
            # Forward pass with automatic mixed precision (fp16)
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass with gradient scaling, step optimizer, and update learning rate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            print("loss:", loss.item())
    
    # Save trained model weights
    filename = dataset_path.split("/")[-1] + model_suffix + ".pth"
    torch.save(model.state_dict(), filename)
    print("Model saved with name:", filename)

def test(dataset_path: str, weights_path: str, model_type: str, dinov3_size: str = "large", se: bool = True, comp_metrics: bool = False, 
         plot_calib_curve: bool = False, visualize_results: bool = False):
    """
    Evaluate a glass segmentation model on test set.
    
    Args:
        dataset_path: Path to dataset root (should contain test/image and test/mask subdirectories)
        weights_path: Path to saved model weights file
        model_type: Type of model - "dinov3", "swin", or "lgnet"
        dinov3_size: Size of DINOv3 backbone if applicable - "large", "base", or "small"
        se: Whether to use Squeeze-Excitation blocks in L+GNet (only relevant for lgnet type)
        comp_metrics: Whether to compute and display metrics (IoU, F-beta, MAE, BER)
        plot_calib_curve: Whether to plot calibration curve
        visualize_results: Whether to save visualization overlays and mask predictions
    """

    # Map dinov3_size to a model name string
    default_dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    if dinov3_size == "large":
        dinov3_model_name = default_dinov3_name
    elif dinov3_size == "base":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vitb")
    elif dinov3_size == "small":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vits")
    else:
        raise ValueError(f"Unsupported dinov3_size: {dinov3_size}")

    # Select model constructor based on argument
    if model_type == "dinov3":
        model = create_mask2former_dinov3_model(ID2LABEL, LABEL2ID, dinov3_model_name)
    elif model_type == "swin":
        model = create_swin_mask2former(ID2LABEL, LABEL2ID)
    else:
        model = create_lgnet(ID2LABEL, LABEL2ID, dinov3_model_name, se=se)

    # Load pre-trained weights and prepare model for evaluation
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    model.to("cuda")

    # Initialize test dataset with image/mask pairs
    test_dataset = SegmentationDataset(
        image_dir= os.path.join(dataset_path, "test/image"),
        mask_dir= os.path.join(dataset_path, "test/mask"),
        processor=processor
    )
    collate_fn = make_collate_fn_test(processor)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize evaluation utilities based on flags
    if comp_metrics:
        metrics = Metrics()
    if plot_calib_curve:
        # Calculate total number of pixels in test set for calibration curve
        total_pixels = compute_pixels_in_dataset(dataset_path)
        calib_curve = CalibCurve(dataset_path, total_pixels)

    if visualize_results:
        visualizer = Visualizer(dataset_path)

    # Perform inference on test set
    with torch.no_grad():
        for i, (batch, images, label_masks, image_ids) in enumerate(test_dataloader):
            # Move batch to GPU and get model predictions
            batch["pixel_values"] = batch["pixel_values"].to("cuda")
            outputs = model(**batch)

            # Extract probability maps from model output
            semantic_probs = m2foutput_to_prob_masks(outputs)

            # Post-process predictions to original image size
            seg = processor.post_process_semantic_segmentation(
                outputs=outputs,
                target_sizes=[x.size[::-1] for x in images])
            
            # Process each prediction in batch
            for j, (pred_mask, image, gt_mask, id) in enumerate(zip(seg, images, label_masks, image_ids)):
                pred_mask = pred_mask.to("cpu")
                gt_mask = gt_mask.to("cpu")

                # Resize probability map to original image size for calibration curve
                prob_resized = F.interpolate(
                    semantic_probs[j, 1].unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                    size=image.size[::-1],   # (height, width)
                    mode="bilinear",
                    align_corners=False
                ).to("cpu").squeeze(0).squeeze(0)

                # Update evaluation metrics if requested
                if comp_metrics:
                    metrics.update(pred_mask, gt_mask)
                if plot_calib_curve:
                    calib_curve.update(prob_resized.flatten().numpy(), gt_mask.flatten().numpy())
                
                # Save visualization if requested
                if visualize_results:
                    visualizer.save(image, pred_mask, id)
    
    # Compute and display final results            
    if comp_metrics:
        metrics.compute()

    if plot_calib_curve:
        calib_curve.plot()

    
    print("Tested with saved model:", weights_path)


def inference_timing(model_type: str, dinov3_size: str = "large", warmup: int = 200, se: bool = True, half_precision: bool=True):
    """
    Benchmark inference speed of a model.
    
    Args:
        model_type: Type of model - "dinov3", "swin", or "lgnet"
        dinov3_size: Size of DINOv3 backbone if applicable - "large", "base", or "small"
        warmup: Number of warmup iterations before timing (to allow GPU to stabilize)
        se: Whether to use Squeeze-Excitation blocks in L+GNet (only relevant for lgnet type)
        half_precision: Whether to run model in fp16 (half precision floating point)
    """

    # Map dinov3_size to a model name string
    default_dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    if dinov3_size == "large":
        dinov3_model_name = default_dinov3_name
    elif dinov3_size == "base":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vitb")
    elif dinov3_size == "small":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vits")
    else:
        raise ValueError(f"Unsupported dinov3_size: {dinov3_size}")

    # Select model constructor based on argument and load with appropriate precision
    if model_type == "dinov3" and half_precision==False:
        model = create_mask2former_dinov3_model(ID2LABEL, LABEL2ID, dinov3_model_name)
    elif model_type == "dinov3" and half_precision==True:
        print("###################HALF PRECISION######################")
        model = create_mask2former_dinov3_model(ID2LABEL, LABEL2ID, dinov3_model_name).half()
    elif model_type == "swin" and half_precision==False:
        model = create_swin_mask2former(ID2LABEL, LABEL2ID)
    elif model_type == "swin" and half_precision==True:
        print("###################HALF PRECISION######################")
        model = create_swin_mask2former(ID2LABEL, LABEL2ID).half()
    elif model_type == "lgnet" and half_precision==False:
        model = create_lgnet(ID2LABEL, LABEL2ID, dinov3_model_name, se=se)
    elif model_type == "lgnet" and half_precision==True:
        print("###################HALF PRECISION######################")
        model = create_lgnet(ID2LABEL, LABEL2ID, dinov3_model_name, se=se).half()

    # Prepare model for evaluation on GPU
    model.eval()
    model.to("cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # Create synthetic input tensor with appropriate precision
        if half_precision:
            batch= {"pixel_values": torch.randint(0, 255, (1, 3, 512, 512)).to(torch.float16)}
        else:
            batch= {"pixel_values": torch.randint(0, 255, (1, 3, 512, 512)).to(torch.float32)}
        batch["pixel_values"] = batch["pixel_values"].to("cuda")
        
        # Run warmup iterations to stabilize GPU state and cache
        time_arr = np.empty(1000)
        for i in range (1000 + warmup):
            if i < warmup:
                # Warmup iterations (not timed)
                outputs = model(**batch)
            else:
                # Timed iterations: use CUDA events for accurate timing
                start.record()
                outputs = model(**batch)
                end.record()
                torch.cuda.synchronize()
                # Store time in seconds (elapsed_time is in milliseconds)
                time_arr[i-warmup] = start.elapsed_time(end)/1000

    # Print timing statistics: full array and frames per second (inverse of mean time)
    print(time_arr)
    print(1/time_arr.mean())



def parse_args():
    """
    Parse command-line arguments for train/test/inference modes.
    
    Returns:
        Parsed arguments namespace with all configuration options
    """
    parser = argparse.ArgumentParser(
        description="Train or test glass segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  Examples:
  # Train the model on a dataset
  python dev.py --mode train --dataset-path /path/to/dataset --suffix v1

  # Test the model with weights
  python dev.py --mode test --dataset-path /path/to/dataset --weights-path checkpoints/model_v1.pth

  # Train and then test
  python dev.py --mode train_and_test --dataset-path /path/to/dataset --suffix v1 --weights-path checkpoints/model_v1.pth
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "train_and_test", "inference_timing"],
        default="train",
        help="Mode: train, test, or train_and_test (default: train)"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Path to dataset root directory. Should contain train/image, train/mask, test/image, test/mask"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for saved model filename during training (e.g., 'v1' -> 'model_v1.pth')"
    )

    parser.add_argument(
        "--weights-path",
        type=str,
        default="",
        help="Path to saved weights for testing (e.g., 'checkpoints/model_v1.pth')"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["dinov3", "swin", "lgnet"],
        default="lgnet",
        help="Model type: dinov3 (create_mask2former_dinov3_model), swin (create_swin_mask2former), lgnet (create_lgnet)"
    )

    parser.add_argument(
        "--dinov3-size",
        type=str,
        choices=["large", "base", "small"],
        default="large",
        help="DINOv3 model size to use when a DINOv3-based model is selected (large, base, small). Default: large"
    )

    parser.add_argument(
        "--no-se",
        action="store_false",
        dest="se",
        default=True,
        help="Disable SE blocks (enabled by default)",
    )

    parser.add_argument(
        "--half-precision",
        action="store_true",
        dest="half_precision",
        default=False,
        help="Run model with half precision floating point (fp16) for inference testing. Disabled by default."
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        dest="metrics",
        default=False,
        help="Report metrics after testing (disabled by default)"
    )

    parser.add_argument(
        "--calib-curve",
        action="store_true",
        dest="calib",
        default=False,
        help="Enable plotting of calibration curve after testing (disabled by default)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        dest="visualize",
        default=False,
        help="Visualize and save predictions during testing (disabled by default)"
    )

    return parser.parse_args()


def main():
    """
    Main entry point: parse command-line arguments and execute requested mode.
    
    Supported modes:
    - train: Train model on dataset and save weights
    - test: Evaluate model on test set with optional metrics/visualization
    - train_and_test: Train model and immediately evaluate on test set
    - inference_timing: Benchmark model inference speed
    """
    args = parse_args()
    
    # Handle training mode
    if args.mode == "train":
        print(f"Running in TRAIN mode")
        print(f"  Dataset: {args.dataset_path}")
        print(f"  Suffix: {args.suffix}")
        print(f"  Model type: {args.model_type}")
        print(f"  DINOv3 size: {args.dinov3_size}")
        print(f"  SE blocks: {args.se}")

        train(args.dataset_path, args.suffix, args.model_type, args.dinov3_size, args.se)

    # Handle testing mode
    elif args.mode == "test":
        print(f"Running in TEST mode")
        print(f"  Dataset: {args.dataset_path}")
        print(f"  Weights: {args.weights_path}")
        print(f"  Model type: {args.model_type}")
        print(f"  DINOv3 size: {args.dinov3_size}")
        print(f"  SE blocks: {args.se}")
        print(f"  Compute metrics: {args.metrics}")
        print(f"  Plot calibration curve: {args.calib}")
        print(f"  Visualize results: {args.visualize}")

        # Validate that weights path is provided for testing
        if not args.weights_path:
            print("ERROR: --weights-path is required for test mode")
            sys.exit(1)
        test(args.dataset_path, args.weights_path, args.model_type, args.dinov3_size, args.se, args.metrics, args.calib, args.visualize)
    
    # Handle training + testing mode
    elif args.mode == "train_and_test":
        print(f"Running TRAIN + TEST mode")
        print(f"  Dataset: {args.dataset_path}")
        print(f"  Suffix: {args.suffix}")
        print(f"  Weights: {args.weights_path}")
        print(f"  Model type: {args.model_type}")
        print(f"  DINOv3 size: {args.dinov3_size}")
        print(f"  SE blocks: {args.se}")

        # First train the model
        train(args.dataset_path, args.suffix, args.model_type, args.dinov3_size, args.se)
        # Then test if weights path provided, otherwise skip test phase
        if args.weights_path:
            test(args.dataset_path, args.weights_path, args.model_type, args.dinov3_size, args.se)
        else:
            print("WARNING: --weights-path not provided, skipping test phase")
    
    # Handle inference timing benchmark mode
    elif args.mode == "inference_timing":
        inference_timing(model_type=args.model_type, dinov3_size=args.dinov3_size, se=args.se, half_precision=args.half_precision)
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
