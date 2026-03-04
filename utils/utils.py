import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
from sklearn.calibration import calibration_curve
import torch
import torch.nn.functional as F
from typing import Dict

from torchmetrics.classification import BinaryJaccardIndex  
from torchmetrics import FBetaScore, MeanAbsoluteError, ConfusionMatrix
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image, to_tensor
import PIL


def compute_pixels_in_dataset(dataset_path: str) -> int:
    total_pixels = 0
    for filename in os.listdir(os.path.join(dataset_path, "test/image")):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
            image_path = os.path.join(dataset_path, "test/image", filename)

            with PIL.Image.open(image_path) as img:
                width, height = img.size
                total_pixels += width * height

    return total_pixels

class Visualizer:
    def __init__(self, dataset_path: str):
        if "GDD" in dataset_path:
            dataset = "GDD"
        elif "GSD" in dataset_path:
            dataset = "GSD"
        elif "HSOD" in dataset_path:
            dataset = "HSOD"
        elif "trans" in dataset_path:
            dataset = "trans10k-stuff"
        else:
            raise ValueError("Unknown dataset for visualization")
        
        self.dataset = dataset

        os.makedirs(f"results_new/{dataset}/overlay", exist_ok=True)
        os.makedirs(f"results_new/{dataset}/mask", exist_ok=True)

    def save(self, image: PIL.Image, pred_mask: torch.Tensor, image_id: str):
        image = to_tensor(image)
        overlay = draw_segmentation_masks(image, pred_mask.bool(), alpha=0.5, colors=["green"])
        overlay = to_pil_image(overlay)
        overlay.save(f"results_new/{self.dataset}/overlay/{image_id}")

        pred_mask_pil = to_pil_image((pred_mask * 255).to(torch.uint8))
        pred_mask_pil.save(f"results_new/{self.dataset}/mask/{image_id}".replace(".jpg",".png"))



class Metrics:
    def __init__(self):
        self.metric_iou = BinaryJaccardIndex()
        self.metric_fb = FBetaScore(task = "binary", beta=np.sqrt(0.3))
        self.metric_mae_bin = MeanAbsoluteError()
        self.metric_cm = ConfusionMatrix(num_classes=2, task="binary")

    def update(self, pred_mask: torch.Tensor, true_mask: torch.Tensor):
        self.metric_iou.update(pred_mask, true_mask)
        self.metric_fb.update(pred_mask, true_mask)
        self.metric_mae_bin.update(pred_mask, true_mask)
        self.metric_cm.update(pred_mask, true_mask)

    def compute(self):
        final_iou = self.metric_iou.compute()
        final_fb = self.metric_fb.compute()
        final_mae_bin = self.metric_mae_bin.compute()
        final_cm = self.metric_cm.compute()
        

        TN, FP = final_cm[0]
        FN, TP = final_cm[1]
        FPR = FP / (FP + TN + 1e-8)
        FNR = FN / (FN + TP + 1e-8)
        BER = 0.5 * (FPR + FNR)

        print("Final IoU:", final_iou)
        print("Final F_beta:", final_fb)
        print("Final MAE binary:", final_mae_bin)
        print("Final BER:", BER)
        

class CalibCurve:
    def __init__(self, dataset_path: str, total_pixels: int = 0, n_bins: int = 10, precomputed: bool = False):
        self.precomputed = precomputed
        self.n_bins = n_bins
        if "GDD" in dataset_path:
            self.title = "Dataset: GDD"
            self.save_path = "GDD_calibration_curve.pdf"
        elif "GSD" in dataset_path:
            self.title = "Dataset: GSD"
            self.save_path = "GSD_calibration_curve.pdf"
        elif "HSOD" in dataset_path:
            self.title = "Dataset: HSO"
            self.save_path = "HSO_calibration_curve.pdf"
        elif "trans" in dataset_path:
            self.title = "Dataset: Trans10k-Stuff"
            self.save_path = "Trans10k_calibration_curve.pdf"
        else:
            raise ValueError("Unknown dataset for calibration curve")
        
        self.probs  = np.empty(total_pixels, dtype=np.float16)
        self.labels = np.empty(total_pixels, dtype=np.bool_)
        # Write pointer
        self.pos = 0

        font_path = "times.ttf"
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rcParams["font.family"] = font_name

    def update(self, probs: np.ndarray, labels: np.ndarray):
        n = probs.size
        self.probs[self.pos:self.pos + n] = probs.astype(np.float16)
        self.labels[self.pos:self.pos + n] = labels.astype(np.bool_)

        self.pos += n

    def plot(self):
        if self.precomputed:
            data = np.load(self.save_path.replace(".pdf", ".npz"))
            prob_true = data["prob_true"]
            prob_pred = data["prob_pred"]
        else:
            prob_true, prob_pred = calibration_curve(self.labels, self.probs, n_bins=self.n_bins)
            np.savez(self.save_path.replace(".pdf", ".npz"), prob_true=prob_true, prob_pred=prob_pred)
            
        plt.figure(figsize=(4, 4))
        plt.plot(prob_pred, prob_true, "o-", label="L+GNet calibration")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05) 
        plt.xlabel("Model confidence")
        plt.ylabel("Fraction of positives")
        plt.title(self.title)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()



def m2foutput_to_prob_masks(outputs: Dict) -> torch.Tensor:
    #--------------------------
    # Function contents copied from https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/mask2former/image_processing_mask2former.py#L988
    # 
    # Copyright 2018- The Hugging Face team. All rights reserved.

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
     
    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Scale back to preprocessed image size - (384, 384) for all models
    masks_queries_logits = torch.nn.functional.interpolate(
        masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
    )

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    #--------------------------
    
    semantic_probs = F.softmax(segmentation, dim=1)  # [B, C, H, W]
 
    return semantic_probs