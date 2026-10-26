"""Whole-body keypoints through rtmlib: a YOLOX person detector trained on
HumanArt and RTMW, both run by onnxruntime. torch is imported first on purpose:
onnxruntime's CUDA provider finds cuDNN through the libraries torch loads."""

import numpy as np
import torch
from rtmlib import Wholebody

import comfy.model_management

# One model per process: loading takes seconds and the node runs once per image.
_model = None

NOSE = 0
EYES = (1, 2)
WRISTS = (9, 10)
ANKLES = (15, 16)
FACE = slice(23, 91)  # the 68 face contour keypoints
HANDS = (slice(91, 112), slice(112, 133))  # the 21 keypoints of the left and right hand
TIPS = (4, 8, 12, 16, 20)  # fingertip index within a hand's 21 keypoints


def keypoints(image):
    """image: (H, W, 3) in [0, 1].

    returns: (P, 133, 2) xy in pixels and (P, 133) scores, one row per person.
    The scores are SimCC maxima, not probabilities: compare them within a person.
    """
    global _model
    if _model is None:
        device = comfy.model_management.get_torch_device().type
        _model = Wholebody(
            mode="performance",
            backend="onnxruntime",
            device="cuda" if device == "cuda" else "cpu",
        )
    rgb = (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return _model(np.ascontiguousarray(rgb[..., ::-1]))
