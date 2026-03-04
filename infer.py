"""Minimal inference demo for glass segmentation with L+GNet model."""

import argparse
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image, to_tensor

from models.LGNet import create_lgnet
from utils.utils import m2foutput_to_prob_masks


def infer(image: Image, weights_path: str, dinov3_size: str = "large"):
    """
    Minimal inference function: process with model, return predictions.
    
    Args:
        image: PIL image
        weights_path: Path to saved model weights
        dinov3_size: DINOv3 size if applicable ("large", "base", or "small")
    
    Returns:
        predicted mask
    """

    # Define label mappings for binary glass segmentation
    ID2LABEL = {0: "background", 1: "glass"}
    LABEL2ID = {"background": 0, "glass": 1}

    # Initialize image processor
    m2f_pretrained = "facebook/mask2former-swin-small-ade-semantic"
    processor = AutoImageProcessor.from_pretrained(m2f_pretrained)
    processor.size = {"height": 512, "width": 512}

    # Map dinov3_size to model name
    default_dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    if dinov3_size == "large":
        dinov3_model_name = default_dinov3_name
    elif dinov3_size == "base":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vitb")
    elif dinov3_size == "small":
        dinov3_model_name = default_dinov3_name.replace("vitl", "vits")
    else:
        raise ValueError(f"Unsupported dinov3_size: {dinov3_size}")

    # Create model
    model = create_lgnet(ID2LABEL, LABEL2ID, dinov3_model_name)

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location="cuda"))
    model.eval()
    model.to("cuda")

    # Preprocess image
    batch = processor(images=[image], return_tensors="pt")
    batch["pixel_values"] = batch["pixel_values"].to("cuda")

    # Inference
    with torch.no_grad():
        outputs = model(**batch)

    pred_mask = processor.post_process_semantic_segmentation(
                outputs=outputs,
                target_sizes=[image.size[::-1]])[0]

    return pred_mask


def main():
    parser = argparse.ArgumentParser(description="Minimal inference demo for L+GNet glass segmentation")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output-folder", type=str, required=True, help="Output folder to save results")
    parser.add_argument("--dinov3-size", type=str, default="large", help="DINOv3 size: large, base, small")
    
    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load image
    image = Image.open(args.image_path).convert("RGB")
    # Run inference
    pred_mask = infer(image, args.weights_path, args.dinov3_size)
    
    # Save predicted mask
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    pred_mask_pil = to_pil_image((pred_mask * 255).to(torch.uint8))
    mask_path = os.path.join(args.output_folder, f"{base_filename}_mask.png")
    pred_mask_pil.save(mask_path)
    print(f"Saved mask to: {mask_path}")
    
    # Save overlay visualization
    overlay = draw_segmentation_masks(to_tensor(image), pred_mask.bool(), alpha=0.5, colors=["green"])
    overlay_pil = to_pil_image(overlay)
    overlay_path = os.path.join(args.output_folder, f"{base_filename}_overlay.png")
    overlay_pil.save(overlay_path)
    print(f"Saved overlay to: {overlay_path}")


if __name__ == "__main__":
    main()
