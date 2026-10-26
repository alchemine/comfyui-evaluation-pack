"""SAM3 text-prompt detection through the copy of SAM3 that ComfyUI-RMBG ships."""

import os
import sys

import torch
from PIL import Image

import folder_paths
import comfy.model_management

# One model per process: loading takes seconds and the node runs once per image.
_processor = None
# The image backbone is most of a detection, and the node asks several prompts
# of one image before moving on: the state of the last image is kept for them.
_last_image = None
_last_state = None


def _rmbg_models_dir():
    for base in folder_paths.get_folder_paths("custom_nodes"):
        path = os.path.join(base, "ComfyUI-RMBG", "models")
        if os.path.isdir(os.path.join(path, "sam3")):
            return path
    raise RuntimeError(
        "ComfyUI-RMBG is not installed; AnatomyEvaluator runs the SAM3 code it ships"
    )


def _load():
    global _processor
    if _processor is None:
        models = _rmbg_models_dir()
        for path in (models, os.path.join(models, "sam3")):
            if path not in sys.path:
                sys.path.insert(0, path)
        # Imported here: the package only exists once ComfyUI-RMBG is on sys.path.
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = os.path.join(folder_paths.models_dir, "sam3", "sam3.pt")
        if not os.path.isfile(checkpoint):
            raise RuntimeError(
                "models/sam3/sam3.pt is missing; run a ComfyUI-RMBG SAM3 node once to download it"
            )
        device = (
            "cuda"
            if comfy.model_management.get_torch_device().type == "cuda"
            else "cpu"
        )
        model = build_sam3_image_model(
            bpe_path=os.path.join(
                models, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
            ),
            device=device,
            eval_mode=True,
            checkpoint_path=checkpoint,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        _processor = Sam3Processor(model, device=device)
    return _processor


def detect(image, prompt, threshold):
    """Masks of every `prompt` instance SAM3 finds in `image` above `threshold`.

    image: (H, W, 3) in [0, 1]
    returns: (K, H, W) bool, on the CPU
    """
    global _last_image, _last_state
    processor = _load()
    if image is not _last_image:
        pil = Image.fromarray(
            (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        )
        _last_image, _last_state = image, processor.set_image(pil)
    processor.set_confidence_threshold(threshold)
    state = processor.set_text_prompt(prompt, dict(_last_state))
    masks = state["masks"]
    if masks.numel() == 0:
        return torch.zeros((0,) + tuple(image.shape[:2]), dtype=torch.bool)
    return masks.flatten(0, 1).bool().cpu()
