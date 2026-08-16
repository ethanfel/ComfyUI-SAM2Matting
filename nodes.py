"""ComfyUI node definitions for temporal SAM2Matting video propagation."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch

import comfy.model_management
import comfy.utils
import folder_paths

from .video_model import (
    MEMORY_MODES,
    SAM3_TEXT_SELECTIONS,
    VARIANT_INFO,
    SAM2MattingVideoModel,
    download_checkpoint,
    make_checkerboard_preview,
)
from .streaming_video import (
    DiskAlphaStore,
    encode_background_video,
    parse_hex_color,
    prepare_tracking_frames,
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


class LoadSAM2MattingVideoPath:
    """Open an existing server-side video as a native ComfyUI VIDEO."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {"default": "/path/to/video.mp4", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load"
    CATEGORY = "SAM2Matting/video"
    DESCRIPTION = (
        "Opens a video that already exists on the machine running ComfyUI. "
        "This avoids the browser upload-size limit and returns a file-backed "
        "native VIDEO without decoding an IMAGE batch."
    )

    @classmethod
    def IS_CHANGED(cls, video_path: str):
        path = Path(video_path).expanduser()
        try:
            stat = path.stat()
        except OSError:
            return float("nan")
        return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, video_path: str):
        from comfy_api.latest import InputImpl

        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Video path does not exist inside the ComfyUI machine/container: "
                f"{path}"
            )
        return (InputImpl.VideoFromFile(str(path)),)


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

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "matte"
    CATEGORY = "SAM2Matting/video"
    DESCRIPTION = (
        "Treats the complete ordered IMAGE batch as one video, seeds the selected "
        "frame with a white-foreground mask, and propagates a single temporal "
        "predictor state in both directions. Returns only the ordered alpha batch "
        "to avoid caching duplicate full-resolution RGB and preview batches."
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
        return (alpha,)


class SAM2MattingVideoBackground:
    """Composite a native VIDEO over a solid color without IMAGE batches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "video": ("VIDEO",),
                "initial_mask": ("MASK",),
                "mask_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1},
                ),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "background_color": (
                    "STRING",
                    {"default": "#808080", "multiline": False},
                ),
                "state_device": (
                    ["gpu", "cpu"],
                    {"default": "gpu"},
                ),
                "output_fps": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01},
                ),
                "crf": (
                    "INT",
                    {"default": 18, "min": 0, "max": 51, "step": 1},
                ),
                "preserve_audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "composite"
    CATEGORY = "SAM2Matting/video"
    DESCRIPTION = (
        "Streams a native ComfyUI VIDEO through temporal matting, composites the "
        "foreground over a solid color, and returns a native H.264 VIDEO. Tracking "
        "frames and mattes are disk-backed, so host RAM does not grow with the "
        "full-resolution clip length. Connect the result to ComfyUI Save Video."
    )

    def composite(
        self,
        model: SAM2MattingVideoModel,
        video,
        initial_mask: torch.Tensor,
        mask_frame: int,
        mask_threshold: float,
        background_color: str,
        state_device: str,
        output_fps: float,
        crf: int,
        preserve_audio: bool,
    ):
        from comfy_api.latest import InputImpl

        if state_device not in ("gpu", "cpu"):
            raise ValueError("state_device must be 'gpu' or 'cpu'")
        background = parse_hex_color(background_color)
        temp_root = folder_paths.get_temp_directory()
        os.makedirs(temp_root, exist_ok=True)
        with NamedTemporaryFile(
            prefix="sam2matting_background_",
            suffix=".mp4",
            dir=temp_root,
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        progress = comfy.utils.ProgressBar(1)
        frames = None
        alphas = None
        try:
            with TemporaryDirectory(
                prefix="sam2matting_stream_",
                dir=temp_root,
            ) as work_directory:
                work_path = Path(work_directory)

                def preprocessing_progress(done: int, total: int) -> None:
                    progress.update_absolute(done, max(total * 3, 1))

                frames, info = prepare_tracking_frames(
                    video=video,
                    path=work_path / "tracking.frames",
                    image_size=int(model.predictor.image_size),
                    kind=model.kind,
                    progress_callback=preprocessing_progress,
                    interrupt_callback=(
                        comfy.model_management.throw_exception_if_processing_interrupted
                    ),
                )
                total_progress = info.frame_count * 3
                alphas = DiskAlphaStore(
                    work_path / "alpha.frames",
                    info.frame_count,
                    info.height,
                    info.width,
                )

                def tracking_progress(done: int, _total: int) -> None:
                    progress.update_absolute(info.frame_count + done, total_progress)

                model.matte_frame_sequence(
                    frames=frames,
                    height=info.height,
                    width=info.width,
                    initial_mask=initial_mask,
                    mask_frame=int(mask_frame),
                    mask_threshold=float(mask_threshold),
                    memory_mode=("balanced" if state_device == "gpu" else "low_vram"),
                    alpha_callback=alphas.write,
                    progress_callback=tracking_progress,
                    interrupt_callback=(
                        comfy.model_management.throw_exception_if_processing_interrupted
                    ),
                )
                alphas.flush()

                def encoding_progress(done: int, _total: int) -> None:
                    progress.update_absolute(
                        info.frame_count * 2 + done,
                        total_progress,
                    )

                encode_background_video(
                    video=video,
                    alpha_store=alphas,
                    info=info,
                    background=background,
                    output_path=output_path,
                    video_only_path=work_path / "composited_video.mp4",
                    output_fps=float(output_fps),
                    crf=int(crf),
                    preserve_audio=bool(preserve_audio),
                    progress_callback=encoding_progress,
                    interrupt_callback=(
                        comfy.model_management.throw_exception_if_processing_interrupted
                    ),
                )
                alphas.close()
                alphas = None
                frames.close()
                frames = None
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
        finally:
            if alphas is not None:
                alphas.close()
            if frames is not None:
                frames.close()

        return (InputImpl.VideoFromFile(str(output_path)),)


class SAM3TextPromptSeedMask:
    """Create one inspectable video seed mask from a SAM3 text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "images": ("IMAGE",),
                "text_prompt": (
                    "STRING",
                    {"default": "person", "multiline": False},
                ),
                "frame_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1},
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "selection": (
                    list(SAM3_TEXT_SELECTIONS),
                    {"default": "highest_score"},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("seed_mask", "preview", "top_score", "mask_frame")
    FUNCTION = "segment"
    CATEGORY = "SAM2Matting/video"
    DESCRIPTION = (
        "Uses the SAM3 detector to find a text-described subject on one frame. "
        "Returns a white-foreground seed mask for SAM2Matting Video, a "
        "checkerboard preview, the highest detection score, and the matching "
        "frame index. The loaded model variant must be sam3."
    )

    def segment(
        self,
        model: SAM2MattingVideoModel,
        images: torch.Tensor,
        text_prompt: str,
        frame_index: int,
        confidence_threshold: float,
        selection: str,
    ):
        progress = comfy.utils.ProgressBar(1)
        seed_mask, top_score = model.text_seed_mask(
            images=images,
            text_prompt=text_prompt,
            frame_index=frame_index,
            confidence_threshold=confidence_threshold,
            selection=selection,
            interrupt_callback=(
                comfy.model_management.throw_exception_if_processing_interrupted
            ),
        )
        frame = images[int(frame_index) : int(frame_index) + 1]
        preview = make_checkerboard_preview(frame, seed_mask)
        progress.update_absolute(1, 1)
        return (seed_mask, preview, float(top_score), int(frame_index))


NODE_CLASS_MAPPINGS = {
    "LoadSAM2MattingVideoPath": LoadSAM2MattingVideoPath,
    "LoadSAM2MattingVideoModel": LoadSAM2MattingVideoModel,
    "SAM3TextPromptSeedMask": SAM3TextPromptSeedMask,
    "SAM2MattingVideo": SAM2MattingVideo,
    "SAM2MattingVideoBackground": SAM2MattingVideoBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM2MattingVideoPath": "Load Video Path (Native)",
    "LoadSAM2MattingVideoModel": "Load SAM2Matting Video Model",
    "SAM3TextPromptSeedMask": "SAM3 Text Prompt to Seed Mask",
    "SAM2MattingVideo": "SAM2Matting Video",
    "SAM2MattingVideoBackground": "SAM2Matting Video Background (Streaming)",
}
