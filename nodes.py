"""ComfyUI node definitions for temporal SAM2Matting video propagation."""

from __future__ import annotations

import os

import torch

import comfy.model_management
import comfy.utils
import folder_paths

from .video_model import (
    MEMORY_MODES,
    VARIANT_INFO,
    SAM2MattingVideoModel,
    download_checkpoint,
    make_checkerboard_preview,
)


MODEL_FOLDER = "sam2matting"
MODEL_TYPE = "SAM2MATTING_VIDEO_MODEL"

if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        MODEL_FOLDER,
        os.path.join(folder_paths.models_dir, MODEL_FOLDER),
    )


def _checkpoint_path(variant: str) -> str:
    filename = VARIANT_INFO[variant]["checkpoint"]
    existing = folder_paths.get_full_path(MODEL_FOLDER, filename)
    if existing is not None:
        return existing
    model_dirs = folder_paths.get_folder_paths(MODEL_FOLDER)
    if not model_dirs:
        raise RuntimeError("ComfyUI did not provide a SAM2Matting model directory")
    return os.path.join(model_dirs[0], filename)


class LoadSAM2MattingVideoModel:
    """Load a reusable official SAM2Matting temporal video predictor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (
                    ["sam2.1_base_plus", "sam2.1_tiny", "sam3"],
                    {"default": "sam2.1_base_plus"},
                ),
                "compile_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "SAM2Matting/video"
    DESCRIPTION = (
        "Loads the official SAM2Matting video predictor. Missing checkpoints are "
        "downloaded to ComfyUI/models/sam2matting. Compilation is optional and "
        "makes the first run substantially slower."
    )

    def load(self, variant: str, compile_model: bool):
        checkpoint_path = _checkpoint_path(variant)
        if not os.path.isfile(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            progress = comfy.utils.ProgressBar(100)

            def report_download(done: int, total: int):
                comfy.model_management.throw_exception_if_processing_interrupted()
                progress.update_absolute(int(done * 100 / total), 100)

            download_checkpoint(variant, checkpoint_path, report_download)

        device = comfy.model_management.get_torch_device()
        model = SAM2MattingVideoModel(
            variant=variant,
            checkpoint_path=checkpoint_path,
            device=device,
            compile_model=compile_model,
        )
        return (model,)


class SAM2MattingVideo:
    """Run one temporally coherent matting state over one IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "images": ("IMAGE",),
                "initial_mask": ("MASK",),
                "mask_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1},
                ),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "memory_mode": (
                    list(MEMORY_MODES),
                    {"default": "balanced"},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("alpha", "foreground_rgb", "preview")
    FUNCTION = "matte"
    CATEGORY = "SAM2Matting/video"
    DESCRIPTION = (
        "Treats the complete ordered IMAGE batch as one video, seeds the selected "
        "frame with a white-foreground mask, and propagates a single temporal "
        "predictor state in both directions. Returns an ordered alpha batch, the "
        "unpremultiplied source RGB, and a checkerboard preview."
    )

    def matte(
        self,
        model: SAM2MattingVideoModel,
        images: torch.Tensor,
        initial_mask: torch.Tensor,
        mask_frame: int,
        mask_threshold: float,
        memory_mode: str,
    ):
        frame_count = int(images.shape[0]) if images.ndim > 0 else 0
        progress = comfy.utils.ProgressBar(max(frame_count, 1))

        def report_progress(done: int, total: int):
            progress.update_absolute(done, total)

        alpha = model.matte_video(
            images=images,
            initial_mask=initial_mask,
            mask_frame=mask_frame,
            mask_threshold=mask_threshold,
            memory_mode=memory_mode,
            progress_callback=report_progress,
            interrupt_callback=(
                comfy.model_management.throw_exception_if_processing_interrupted
            ),
        )
        foreground_rgb = images.detach().float().cpu()[..., :3].clamp(0.0, 1.0)
        preview = make_checkerboard_preview(foreground_rgb, alpha)
        return (alpha, foreground_rgb, preview)


NODE_CLASS_MAPPINGS = {
    "LoadSAM2MattingVideoModel": LoadSAM2MattingVideoModel,
    "SAM2MattingVideo": SAM2MattingVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM2MattingVideoModel": "Load SAM2Matting Video Model",
    "SAM2MattingVideo": "SAM2Matting Video",
}
