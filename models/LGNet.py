# Copyright 2025 [Chan Ho Bae / GitHub @Carti-97]
# Parts of contents of this file have been copied / derived from the work of the above credited person.
# https://github.com/Carti-97/DINOv3-Mask2former 
# Especially classes Adapter, DinoV3WithAdapterBackbone and functions create_mask2former_dinov3_model, get_model_info.

# Copyright 2026 [Risto Ojala / GitHub @ojalar]

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

import torch
import torch.nn as nn
from typing import List, Dict, Union
from transformers import AutoModel, AutoModelForUniversalSegmentation
from transformers.utils import ModelOutput
from dataclasses import dataclass
import logging
from torchvision.ops import SqueezeExcitation
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BackboneOutput(ModelOutput):
    """
    Custom output class for the DINOv3 backbone to be compatible with Mask2Former.
    """
    feature_maps: List[torch.Tensor]

class Adapter(nn.Module):
    """
    Adapter module to convert DINOv3 features to expected channels for Mask2Former head.
    """
    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.projections[i](feat) for i, feat in enumerate(features)]


class DinoV3WithAdapterBackbone(nn.Module):
    """
    Custom backbone that combines DINOv3 (available in large/base/small variants) with adapter layers for Mask2Former compatibility.
    Extracts features from intermediate layers and projects them to match Mask2Former channel dimensions.
    """
    def __init__(self, model_name: str, out_channels: List[int], standalone: bool = True, dinov3_variant: str = "large"):
        super().__init__()
        # Load pre-trained DINOv3 model from HuggingFace
        self.model = AutoModel.from_pretrained(model_name)
        # Create adapter to convert DINOv3 hidden size to target channel dimensions
        self.adapter = Adapter(self.model.config.hidden_size, out_channels)
        self.standalone = standalone
        
        # Define output features for Mask2Former compatibility
        self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
        self._out_feature_channels = {name: ch for name, ch in zip(self.out_features, out_channels)}
        # Feature strides correspond to spatial downsampling factors at each stage
        self._out_feature_strides = {"stage_0": 4, "stage_1": 8, "stage_2": 16, "stage_3": 32}
        
        # Select which transformer layers to extract features from, depends on model size
        # Layers to extract from DINOv3
        if dinov3_variant == "large":
            self.layers_to_extract = [5, 11, 17, 23]  # Adjusted for Large model (24 layers total)
        elif dinov3_variant == "base" or dinov3_variant == "small":
            self.layers_to_extract = [2, 5, 8, 11]  # Adjusted for Base and Small model (12 layers total)
        else:
            raise ValueError(f"Unsupported dinov3_variant: {dinov3_variant}")
    
    def forward(self, x: torch.Tensor) -> Union[BackboneOutput[torch.Tensor], List[torch.Tensor]]:
        # Pass input through DINOv3 model and retrieve all hidden states for intermediate layer extraction
        # Get DINOv3 outputs with all hidden states
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        
        # Calculate spatial dimensions after patch embedding
        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        num_reg_tokens = self.model.config.num_register_tokens
        
        # Extract features from selected intermediate layers for feature pyramid
        # Extract features from different layers
        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[layer_idx + 1]  # Skip CLS token
            # Reshape from (B, N, C) to (B, C, H, W)
            feature_map = layer_output[:, num_reg_tokens+1:, :].permute(0, 2, 1).reshape(
                batch_size, self.model.config.hidden_size, patch_height, patch_width
            )
            extracted_features.append(feature_map)
        
        # Project all features to target channel dimensions using 1x1 convolutions
        # Apply adapter to convert channels
        adapted_features = self.adapter(extracted_features)
        
        # Return adapted features either wrapped in BackboneOutput (for standalone use) or as raw list
        if self.standalone:
            # Return features with proper naming for Mask2Former
            return BackboneOutput(
                feature_maps = adapted_features
                )
            #return {name: feat for name, feat in zip(self.out_features, adapted_features)}
        else:
            # Return raw feature list for integration with other backbones
            return extracted_features

class SEChannelReduction(nn.Module):
    """Channel reduction module with Squeeze-Excitation attention for feature refinement.
    Projects input channels to output channels using convolution with SE attention and residual connection.
    """
    def __init__(self, in_ch: int, out_ch: int, se_ratio: int = 8):
        super().__init__()
        # Intermediate channel (at least half of input or equal to output)
        intrmd_ch = np.max((in_ch//2, out_ch))
        self.se_block = nn.Sequential(
            nn.Conv2d(in_ch, intrmd_ch, 3, padding=1),
            nn.BatchNorm2d(intrmd_ch),
            nn.ReLU(),
            nn.Conv2d(intrmd_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            SqueezeExcitation(out_ch, out_ch // se_ratio),
        )
        # Skip connection via 1x1 convolution to project input to output channels
        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, features: torch.Tensor):
        # Combine main path with residual skip connection
        return self.se_block(features) + self.skip_conv(features)

class DualBackboneAdapter(nn.Module):
    """Adapter module to fuse features from learned and general backbones.
    Handles spatial resolution alignment and optionally applies Squeeze-Excitation attention for feature fusion.
    """
    def __init__(self, out_channels: List[int], out_strides: Dict[str, int], in_channels: int, se: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.out_strides = out_strides
        # Hardcoded ViT-stride for DINOv3 (ViT-based model)
        vit_stride = 16
        # Calculate resize ratios for each pyramid level relative to ViT stride
        resize_ratios = [vit_stride / stride for stride in out_strides.values()]
        
        # Create resizing operations for each feature level to align spatial resolutions
        self.resizing = nn.ModuleList()
        for i, out_ch in enumerate(out_channels):
            if resize_ratios[i] > 1:
                # Upsample when target stride is smaller (higher resolution needed)
                self.resizing.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=resize_ratios[i], mode='bilinear', align_corners=False),
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
                    )
                )
            elif resize_ratios[i] == 1:
                # No resizing needed if strides match
                self.resizing.append(nn.Identity())
            else:
                # Downsample when target stride is larger (lower resolution needed)
                self.resizing.append(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=int(1/resize_ratios[i]), padding=1)
                )
        
        # Create projection modules: either with SE channel reduction for better feature fusion or simple 1x1 convolution
        if se:
            # Use SE blocks for channel-aware feature fusion with attention
            self.projections = nn.ModuleList([
                SEChannelReduction(out_ch + in_channels, out_ch) for out_ch in out_channels
            ]) 
        else:
            # Simple channel projection via 1x1 convolution
            self.projections = nn.ModuleList([
                nn.Conv2d(in_channels + out_ch, out_ch, kernel_size=1) for out_ch in out_channels
            ])
        
    def forward(self, l_features: List[torch.Tensor], g_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fuse learned and general backbone features through spatial alignment and channel projection."""
        final_features = []
        for i, l_feat in enumerate(l_features):
            # Resize general features to match learned features spatial dimensions
            resized_g_feat = self.resizing[i](g_features[i])
            # Concatenate learned and resized general features along channel dimension
            combined_feat = torch.cat([l_feat, resized_g_feat], dim=1)
            # Project combined features to target channel dimension (with optional SE attention)
            projected_feat = self.projections[i](combined_feat)
            final_features.append(projected_feat)
        return final_features

class LGBackbone(nn.Module):
    """Combined backbone fusing Learned (task-specific) and General (foundation model) features.
    The L+G architecture leverages both task-specific learning and general context.
    """
    def __init__(self, learned_backbone: nn.Module, general_backbone: nn.Module, out_channels: List[int], se: bool = True):
        super().__init__()
        # Learned backbone: Swin-S
        self.l_backbone = learned_backbone
        # General backbone: DINOv3
        self.g_backbone = general_backbone

        # Define output features for Mask2Former compatibility
        self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
        self._out_feature_channels = {name: ch for name, ch in zip(self.out_features, out_channels)}
        # Feature pyramid strides: coarser features at higher stages
        self._out_feature_strides = {"stage_0": 4, "stage_1": 8, "stage_2": 16, "stage_3": 32}

        # Adapter for fusing learned and general backbone outputs
        self.adapter = DualBackboneAdapter(out_channels, self._out_feature_strides, self.g_backbone.model.config.hidden_size, se=se)

    def forward(self, x: torch.Tensor) -> BackboneOutput[torch.Tensor]:
        """Process input through both backbones and fuse their features."""
        # Extract features from learned backbone
        l_features = self.l_backbone(x)
        # Extract features from general backbone (foundation model)
        g_features = self.g_backbone(x)
        # Fuse learned and general features through adapter (handles spatial alignment and projection)
        final_features = self.adapter(l_features.feature_maps, g_features)

        # Return fused features wrapped in BackboneOutput for Mask2Former compatibility
        return BackboneOutput(
                feature_maps = final_features
                )
        #return {name: feat for name, feat in zip(self.out_features, final_features)}

def create_swin_mask2former(
    label2id: Dict[str, int],
    id2label: Dict[int, str]
) -> AutoModelForUniversalSegmentation:
    """Create a Mask2Former model with Swin-Small backbone (baseline architecture).
    
    Args:
        label2id: Mapping from label names to class indices
        id2label: Mapping from class indices to label names
    
    Returns:
        Initialized Mask2Former model with Swin-Small encoder
    """
    mask2former_model_name = "facebook/mask2former-swin-small-ade-semantic"
    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    return model

def create_lgnet(
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    dinov3_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    se: bool = True,
    expected_channels: List[int] = [96, 192, 384, 768], # swin small config
    freeze_backbone: bool = True,
    hub_token: str = None
) -> AutoModelForUniversalSegmentation:
    """Create an L+GNet model: Mask2Former with fused Swin (learned) and DINOv3 (general) backbones.
    
    Args:
        label2id: Mapping from label names to class indices
        id2label: Mapping from class indices to label names
        dinov3_model_name: HuggingFace model identifier for DINOv3 (vitl/vitb/vits)
        se: Whether to use Squeeze-Excitation attention in feature fusion
        expected_channels: Channel dimensions for each stage [96, 192, 384, 768] for Swin-Small
        freeze_backbone: Whether to freeze the general backbone (DINOv3) parameters
        hub_token: HuggingFace API token for accessing gated models
    
    Returns:
        Initialized L+GNet model with fused learned and general backbones
    """
    # Fixed Mask2Former base model - using small version
    mask2former_model_name = "facebook/mask2former-swin-small-ade-semantic"
    
    # 1. Load the base Mask2Former model with Swin-Small encoder (learned backbone)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=hub_token,
    )
    
    # 2. Create custom DINOv3 backbone with adapter (general backbone for rich semantic features)
    # Detect DINOv3 variant size from model name (vitl=large, vitb=base, vits=small)
    if "vitl" in dinov3_model_name:
        dinov3_variant = "large"
    elif "vitb" in dinov3_model_name:
        dinov3_variant = "base"
    elif "vits" in dinov3_model_name:
        dinov3_variant = "small"
    else:
        raise ValueError(f"Unsupported dinov3_model_name: {dinov3_model_name}")
    
    # Initialize DINOv3 backbone with adapter for feature extraction
    dinov3_backbone = DinoV3WithAdapterBackbone(dinov3_model_name, expected_channels, standalone=False, dinov3_variant=dinov3_variant)

    # 3. Replace the Mask2Former encoder with L+G backbone that fuses learned and general features
    model.model.pixel_level_module.encoder = LGBackbone(model.model.pixel_level_module.encoder, dinov3_backbone, expected_channels, se=se)

    # Freeze the general backbone (DINOv3) parameters to leverage pre-trained knowledge without fine-tuning
    for param in model.model.pixel_level_module.encoder.g_backbone.parameters():
        param.requires_grad = False
    
    return model

def create_mask2former_dinov3_model(
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    dinov3_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    expected_channels: List[int] = [96, 192, 384, 768], #swin small config
    freeze_backbone: bool = True,
    hub_token: str = None
) -> AutoModelForUniversalSegmentation:
    """Create a Mask2Former model with DINOv3 backbone (DINOv3-only variant without learned backbone fusion).
    
    Args:
        label2id: Mapping from label names to class indices
        id2label: Mapping from class indices to label names
        dinov3_model_name: HuggingFace model identifier for DINOv3 (vitl/vitb/vits)
        expected_channels: Channel dimensions for each stage
        freeze_backbone: Whether to freeze DINOv3 parameters
        hub_token: HuggingFace API token for accessing gated models
    
    Returns:
        Initialized Mask2Former model with DINOv3 encoder (backbone only, no fusion)
    """
    # Fixed Mask2Former base model - using small version
    mask2former_model_name = "facebook/mask2former-swin-small-ade-semantic"
    
    # 1. Load the base Mask2Former-Small model
    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=hub_token,
    )
    
    # 2. Create custom DINOv3 backbone with adapter (standalone mode, no fusion)
    # Detect DINOv3 variant from model name
    if "vitl" in dinov3_model_name:
        dinov3_variant = "large"
    elif "vitb" in dinov3_model_name:
        dinov3_variant = "base"
    elif "vits" in dinov3_model_name:
        dinov3_variant = "small"
    else:
        raise ValueError(f"Unsupported dinov3_model_name: {dinov3_model_name}")
    
    # Create DINOv3 backbone in standalone mode
    custom_backbone = DinoV3WithAdapterBackbone(dinov3_model_name, expected_channels, dinov3_variant=dinov3_variant)
    
    # 3. Replace Mask2Former encoder with DINOv3 backbone
    model.model.pixel_level_module.encoder = custom_backbone

    # Freeze all DINOv3 backbone parameters (pre-trained weights, no further training)
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad = False
    
    logger.info("Successfully created DINOv3 Mask2Former model.")
    
    return model


def get_model_info(model: AutoModelForUniversalSegmentation) -> Dict:
    """Compute and return model parameter statistics.
    
    Args:
        model: PyTorch model to analyze
    
    Returns:
        Dictionary containing total and trainable parameter counts
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Count trainable parameters (requires_grad=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }



if __name__ == "__main__":
    # Define class labels for glass segmentation task
    ID2LABEL = {
        0: "background",
        1: "glass"
    }

    LABEL2ID = {
        "background": 0,
        "glass": 1
    }

    # Test 1: DINOv3-Large with Mask2Former (standalone DINOv3)
    model = create_mask2former_dinov3_model(ID2LABEL, LABEL2ID, "facebook/dinov3-vitl16-pretrain-lvd1689m")
    print("DINOv3-L M2F parameters:", get_model_info(model))
    
    # Test 2: Swin-Small with Mask2Former (baseline learned backbone)
    model = create_swin_mask2former(ID2LABEL, LABEL2ID)
    print("SWIN-S M2F parameters:", get_model_info(model))

    # Test 3: L+GNet with DINOv3-Base (learned + general backbone fusion)
    model = create_lgnet(ID2LABEL, LABEL2ID, "facebook/dinov3-vitb16-pretrain-lvd1689m")
    print("L+GNet DINOv3-B parameters:", get_model_info(model))

    # Test 4: L+GNet with DINOv3-Small
    model = create_lgnet(ID2LABEL, LABEL2ID, "facebook/dinov3-vits16-pretrain-lvd1689m")
    print("L+GNet DINOv3-S parameters:", get_model_info(model))

    # Test 5: L+GNet without SE attention (simpler fusion)
    model = create_lgnet(ID2LABEL, LABEL2ID, "facebook/dinov3-vitl16-pretrain-lvd1689m", se=False)
    print("L+GNet DINOv3-L (no SE) parameters:", get_model_info(model))

    # Test 6: L+GNet with DINOv3-Large (full-scale model with SE channel reduction)
    model = create_lgnet(ID2LABEL, LABEL2ID, "facebook/dinov3-vitl16-pretrain-lvd1689m")
    print("L+GNet DINOv3-L parameters:", get_model_info(model))

    # Print component-level parameter breakdown for L+GNet
    print("\n--- Component Breakdown ---")
    print("SWIN-S (learned backbone) parameters:", get_model_info(model.model.pixel_level_module.encoder.l_backbone))
    print("DINOv3-L (general backbone) parameters:", get_model_info(model.model.pixel_level_module.encoder.g_backbone))
    print("SE Channel reduction (projections) parameters:", get_model_info(model.model.pixel_level_module.encoder.adapter.projections))
    print("Spatial resizing adapters parameters:", get_model_info(model.model.pixel_level_module.encoder.adapter.resizing))
