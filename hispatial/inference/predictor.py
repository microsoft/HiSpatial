"""HiSpatial inference predictor -- loads a trained checkpoint and answers spatial queries."""

import os
import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 448

def _resolve_backbone(backbone_path=None):
    """Return a backbone path, preferring a local copy if available."""
    if backbone_path is not None:
        return backbone_path

    return 'google/paligemma2-3b-pt-448'


class HiSpatialPredictor:
    """Load a trained HiSpatialVLM checkpoint and run spatial VQA inference.

    Args:
        model_load_path: Path to the ``weights.pt`` checkpoint file.
        gpu_rank: CUDA device index.
        backbone_path: Optional explicit path to the PaliGemma2 backbone.
    """

    def __init__(self, model_load_path, gpu_rank=0, backbone_path=None, **kwargs):
        self.device = torch.device(f"cuda:{gpu_rank}")
        self.img_size = IMAGE_SIZE

        if isinstance(model_load_path, list):
            model_load_path = model_load_path[0]
        self.model_load_path = model_load_path

        from transformers import PaliGemmaProcessor
        from hispatial.model import HiSpatialVLM

        self.backbone_path = _resolve_backbone(backbone_path)

        self.model = HiSpatialVLM.from_pretrained(
            pretrained_model_name_or_path=self.backbone_path,
            attn_implementation="eager",
        )

        checkpoint = torch.load(self.model_load_path, map_location="cpu", weights_only=False)
        result = self.model.load_state_dict(checkpoint, strict=False)
        if result.missing_keys:
            print(f"Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys}")

        self.model.eval()
        self.model.to(self.device)
        self.processor = PaliGemmaProcessor.from_pretrained(self.backbone_path)

    def query(self, image, prompt, xyz_dict=None, xyz_values=None):
        """Run a spatial VQA query on an image.

        Args:
            image: Input image (file path, PIL Image, or numpy array).
            prompt: Text prompt for the model.
            xyz_dict: Optional dict with ``mask`` and ``points`` tensors from MoGe.
            xyz_values: Optional precomputed [4, H, W] tensor.

        Returns:
            Decoded text response from the model.
        """
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        I2 = cv2.resize(image, (self.img_size, self.img_size))

        if xyz_values is None and xyz_dict is not None:
            mask = xyz_dict["mask"].cpu().numpy().astype(np.int8)
            xyz = xyz_dict["points"].cpu().numpy()

            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            xyz = cv2.resize(xyz, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            mask = torch.from_numpy(mask)
            xyz = torch.from_numpy(xyz)

            xyz[xyz > 250] = 250
            xyz[torch.isinf(xyz)] = 250
            xyz[torch.isnan(xyz)] = 250

            xyz_values = xyz.permute(2, 0, 1)
            xyz_values = torch.cat((xyz_values, mask.unsqueeze(0)), dim=0)

        if "<image>" not in prompt:
            prompt = "<image>" + prompt

        inputs = self.processor(text=prompt, images=I2, return_tensors="pt")
        inputs["xyz_values"] = xyz_values.unsqueeze(0)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            inputs = inputs.to(self.device)
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=False)

        return decoded
