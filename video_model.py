"""Runtime adapter for temporally propagated SAM2Matting video inference.

This module deliberately has no ComfyUI imports.  It turns a ComfyUI-style
``[frames, height, width, channels]`` tensor into the in-memory frame sequence
expected by the vendored SAM2Matting predictors and normalizes the differing
SAM2/SAM3 propagation outputs into one ordered alpha batch.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "vendor"
SAM2_VENDOR_PACKAGE = "comfyui_sam2matting_sam2"
SAM3_VENDOR_PACKAGE = "comfyui_sam2matting_sam3"

VARIANT_INFO = {
    "sam2.1_base_plus": {
        "kind": "sam2",
        "config": "configs/sam2matting-sam2.1base+.yaml",
        "checkpoint": "SAM2Matting-SAM2.1Base+.pt",
        "url": "https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints/SAM2Matting-SAM2.1Base%2B.pt",
    },
    "sam2.1_tiny": {
        "kind": "sam2",
        "config": "configs/sam2matting-sam2.1tiny.yaml",
        "checkpoint": "SAM2Matting-SAM2.1Tiny.pt",
        "url": "https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints/SAM2Matting-SAM2.1Tiny.pt",
    },
    "sam3": {
        "kind": "sam3",
        "checkpoint": "SAM2Matting-SAM3.pt",
        "url": "https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints/SAM2Matting-SAM3.pt",
    },
}

MEMORY_MODES = {
    "balanced": (True, False),
    "low_vram": (True, True),
    "maximum_speed": (False, False),
}

SAM3_TEXT_SELECTIONS = ("highest_score", "combine_all")

ProgressCallback = Callable[[int, int], None]
InterruptCallback = Callable[[], None]
AlphaCallback = Callable[[int, torch.Tensor], None]


def download_checkpoint(
    variant: str,
    destination: str | os.PathLike[str],
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Atomically download a checkpoint, with a unique partial file per run."""
    if variant not in VARIANT_INFO:
        raise ValueError(f"Unknown SAM2Matting variant: {variant}")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        VARIANT_INFO[variant]["url"],
        headers={"User-Agent": "ComfyUI-SAM2Matting-video"},
    )
    partial_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as partial:
            partial_path = Path(partial.name)
            with urllib.request.urlopen(request) as response:
                total = int(response.headers.get("Content-Length", 0))
                done = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    partial.write(chunk)
                    done += len(chunk)
                    if progress_callback is not None and total > 0:
                        progress_callback(done, total)
                if total > 0 and done != total:
                    raise IOError(
                        f"Incomplete checkpoint download: received {done} of {total} bytes"
                    )
        os.replace(partial_path, destination)
    except BaseException:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise


def _activate_vendored_package(package_name: str) -> None:
    """Put the official vendored package first, refusing unsafe name clashes."""
    vendor_path = str(VENDOR_DIR)
    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "")).resolve()
        try:
            existing_file.relative_to(VENDOR_DIR)
        except ValueError as exc:
            raise RuntimeError(
                f"A different top-level '{package_name}' package is already loaded "
                f"from {existing_file}. Restart ComfyUI with only one {package_name} "
                "implementation enabled to avoid mixing incompatible model code."
            ) from exc
    # Other custom nodes can insert paths ahead of ours after startup. Merely
    # checking for membership is therefore insufficient: an unrelated package
    # with the same import name can still win. Keep one authoritative entry at
    # the front and invalidate cached import-directory listings.
    sys.path[:] = [entry for entry in sys.path if entry != vendor_path]
    sys.path.insert(0, vendor_path)
    importlib.invalidate_caches()


def _device_from_predictor(predictor, fallback: torch.device) -> torch.device:
    value = getattr(predictor, "device", fallback)
    return value if isinstance(value, torch.device) else torch.device(value)


def _move_cached_tensors(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_cached_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_cached_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_cached_tensors(item, device) for item in value]
    return value


def _move_module_and_tensor_caches(module, device: str | torch.device) -> None:
    """Move a module plus SAM3's unregistered positional/coordinate caches."""
    destination = torch.device(device)
    module.to(destination)
    for child in module.modules():
        for attribute in ("cache", "coord_cache", "compilable_cord_cache"):
            if hasattr(child, attribute):
                setattr(
                    child,
                    attribute,
                    _move_cached_tensors(getattr(child, attribute), destination),
                )


def _resize_and_normalize_frame(
    frame_hwc: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    destination: torch.device,
) -> torch.Tensor:
    frame = frame_hwc[..., :3].detach().to(device=destination, dtype=torch.float32)
    frame = frame.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
    frame = F.interpolate(
        frame,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    mean_tensor = frame.new_tensor(mean)[:, None, None]
    std_tensor = frame.new_tensor(std)[:, None, None]
    return (frame - mean_tensor) / std_tensor


class _LazyFrameSequence:
    """Resize/normalize one CPU frame on demand instead of duplicating a clip."""

    def __init__(
        self,
        images: torch.Tensor,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.images = images.detach().cpu()
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return _resize_and_normalize_frame(
            self.images[index],
            self.image_size,
            self.mean,
            self.std,
            torch.device("cpu"),
        )


def _prepare_frames(
    images: torch.Tensor,
    image_size: int,
    kind: str,
    device: torch.device,
    offload_video_to_cpu: bool,
    interrupt_callback: InterruptCallback | None = None,
):
    mean, std = frame_normalization(kind)

    if offload_video_to_cpu:
        return _LazyFrameSequence(images, image_size, mean, std)

    prepared = []
    for frame in images:
        if interrupt_callback is not None:
            interrupt_callback()
        prepared.append(
            _resize_and_normalize_frame(frame, image_size, mean, std, device)
        )
    return torch.stack(prepared, dim=0)


def frame_normalization(kind: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the normalization used by the selected temporal backbone."""
    if kind == "sam2":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if kind == "sam3":
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    raise ValueError(f"Unknown predictor kind: {kind}")


def prepare_seed_mask(
    mask: torch.Tensor,
    mask_frame: int,
    frame_count: int,
    height: int,
    width: int,
    threshold: float,
) -> torch.Tensor:
    """Select, resize, and binarize a ComfyUI mask; white is foreground."""
    mask = torch.as_tensor(mask).detach().float().cpu()
    while mask.ndim > 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    while mask.ndim > 3 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    if mask.ndim == 2:
        selected = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            selected = mask[0]
        elif mask.shape[0] == frame_count:
            selected = mask[mask_frame]
        else:
            raise ValueError(
                "initial_mask must contain either one mask or one mask per input "
                f"frame; got {mask.shape[0]} masks for {frame_count} frames"
            )
    else:
        raise ValueError(
            f"initial_mask must have shape [H,W] or [B,H,W], got {tuple(mask.shape)}"
        )

    selected = F.interpolate(
        selected[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    selected = (selected >= float(threshold)).float()
    if not bool(selected.any()):
        raise ValueError(
            "The selected initial mask contains no foreground pixels after "
            f"applying mask_threshold={threshold}. White must mean foreground."
        )
    return selected


def select_sam3_text_mask(
    masks,
    scores,
    selection: str,
) -> tuple[torch.Tensor, float]:
    """Reduce SAM3 text detections to one white-foreground seed mask."""
    if selection not in SAM3_TEXT_SELECTIONS:
        raise ValueError(
            f"Unknown SAM3 text selection {selection!r}; choose one of "
            f"{SAM3_TEXT_SELECTIONS}"
        )

    masks_tensor = torch.as_tensor(masks).detach()
    if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
        masks_tensor = masks_tensor[:, 0]
    elif masks_tensor.ndim == 2:
        masks_tensor = masks_tensor.unsqueeze(0)
    if masks_tensor.ndim != 3:
        raise RuntimeError(
            "SAM3 returned masks with an unsupported shape: "
            f"{tuple(masks_tensor.shape)}"
        )

    scores_tensor = torch.as_tensor(scores).detach().float().flatten()
    if masks_tensor.shape[0] == 0 or scores_tensor.numel() == 0:
        raise RuntimeError("SAM3 did not find an object above the confidence threshold")
    if masks_tensor.shape[0] != scores_tensor.numel():
        raise RuntimeError(
            "SAM3 returned a different number of masks and scores: "
            f"{masks_tensor.shape[0]} masks, {scores_tensor.numel()} scores"
        )

    top_index = int(torch.argmax(scores_tensor).item())
    top_score = float(scores_tensor[top_index].item())
    if selection == "highest_score":
        selected = masks_tensor[top_index]
    else:
        selected = masks_tensor.any(dim=0)
    return selected.float().cpu().clamp(0.0, 1.0), top_score


def _init_sam2_state(
    predictor,
    frames,
    height: int,
    width: int,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
):
    """Equivalent to upstream init_state(), but backed by in-memory tensors."""
    compute_device = _device_from_predictor(predictor, torch.device("cpu"))
    state = {
        "images": frames,
        "num_frames": len(frames),
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
        "video_height": height,
        "video_width": width,
        "device": compute_device,
        "storage_device": torch.device("cpu")
        if offload_state_to_cpu
        else compute_device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    return state


def _init_sam3_state(
    predictor,
    frames,
    height: int,
    width: int,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
):
    state = predictor.init_state(
        video_height=height,
        video_width=width,
        num_frames=len(frames),
        video_path=None,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
    )
    state["images"] = frames
    # Upstream currently hard-codes the default CUDA device in two places.
    # Correct the state so ComfyUI's selected CUDA device remains authoritative.
    state["device"] = _device_from_predictor(predictor, torch.device("cuda"))
    if not offload_state_to_cpu:
        state["storage_device"] = state["device"]
    return state


def _extract_propagated_alpha(result) -> tuple[int, object]:
    """Handle the upstream 5-item SAM2 and 6-item SAM3 result tuples."""
    if not isinstance(result, (tuple, list)) or len(result) not in (5, 6):
        raise RuntimeError(
            "Unexpected SAM2Matting propagation result; expected a 5-item SAM2 "
            f"or 6-item SAM3 tuple, got {type(result).__name__}"
        )
    return int(result[0]), result[-2]


def _alpha_to_2d(alpha, height: int, width: int) -> torch.Tensor:
    alpha_tensor = torch.as_tensor(
        np.asarray(alpha) if isinstance(alpha, np.ndarray) else alpha
    )
    alpha_tensor = alpha_tensor.detach().float().cpu().squeeze()
    if alpha_tensor.ndim != 2:
        raise RuntimeError(
            f"Expected a single-object 2-D alpha matte, got {tuple(alpha_tensor.shape)}"
        )
    if alpha_tensor.shape != (height, width):
        alpha_tensor = F.interpolate(
            alpha_tensor[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return alpha_tensor.clamp(0.0, 1.0)


def _propagation_passes(
    predictor,
    state,
    mask_frame: int,
) -> Iterator[tuple[bool, tuple]]:
    for result in predictor.propagate_in_video(
        state,
        start_frame_idx=mask_frame,
        max_frame_num_to_track=state["num_frames"],
        reverse=False,
        tqdm_disable=True,
    ):
        yield False, result
    if mask_frame > 0:
        for result in predictor.propagate_in_video(
            state,
            start_frame_idx=mask_frame,
            max_frame_num_to_track=state["num_frames"],
            reverse=True,
            tqdm_disable=True,
        ):
            yield True, result


def _temporal_state_horizon(predictor) -> int:
    """Return the furthest non-conditioning frame the predictor can attend to."""
    num_maskmem = max(int(getattr(predictor, "num_maskmem", 0)), 0)
    stride = max(int(getattr(predictor, "memory_temporal_stride_for_eval", 1)), 1)
    memory_horizon = 0 if num_maskmem < 2 else (num_maskmem - 2) * stride + 1
    pointer_horizon = max(
        int(getattr(predictor, "max_obj_ptrs_in_encoder", 1)) - 1,
        0,
    )
    return max(memory_horizon, pointer_horizon, 1)


def _prune_temporal_state(
    predictor,
    state: dict,
    *,
    current_frame: int,
    mask_frame: int,
    reverse: bool,
) -> None:
    """Discard non-conditioning outputs outside the model's attention window.

    During the forward pass, the first window after a non-zero seed is retained
    as well as the moving recent window. The reverse pass can consequently see
    the same seed-near forward context as the unbounded upstream implementation.
    """
    horizon = _temporal_state_horizon(predictor)

    def should_remove(frame_index: int) -> bool:
        if reverse:
            return frame_index > current_frame + horizon
        if frame_index >= current_frame - horizon:
            return False
        preserve_for_reverse = mask_frame > 0 and frame_index <= mask_frame + horizon
        return not preserve_for_reverse

    removed: set[int] = set()
    output_dict = state.get("output_dict")
    if isinstance(output_dict, dict):
        non_cond = output_dict.get("non_cond_frame_outputs", {})
        for frame_index in tuple(non_cond):
            if should_remove(int(frame_index)):
                non_cond.pop(frame_index, None)
                removed.add(int(frame_index))

    for obj_output_dict in state.get("output_dict_per_obj", {}).values():
        non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
        for frame_index in tuple(non_cond):
            if should_remove(int(frame_index)):
                non_cond.pop(frame_index, None)
                removed.add(int(frame_index))

    if not removed:
        return

    frames_already_tracked = state.get("frames_already_tracked")
    if isinstance(frames_already_tracked, dict):
        for frame_index in removed:
            frames_already_tracked.pop(frame_index, None)

    for tracked_frames in state.get("frames_tracked_per_obj", {}).values():
        for frame_index in removed:
            tracked_frames.pop(frame_index, None)


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16
    try:
        if not torch.cuda.is_bf16_supported():
            dtype = torch.float16
    except (AttributeError, RuntimeError):
        dtype = torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class SAM2MattingVideoModel:
    """One loaded predictor, reusable across independent ComfyUI clip runs."""

    def __init__(
        self,
        variant: str,
        checkpoint_path: str | os.PathLike[str],
        device: str | torch.device,
        compile_model: bool = False,
    ) -> None:
        if variant not in VARIANT_INFO:
            raise ValueError(f"Unknown SAM2Matting variant: {variant}")
        self.variant = variant
        self.kind = VARIANT_INFO[variant]["kind"]
        self.device = torch.device(device)
        self.compile_model = bool(compile_model)
        self.checkpoint_path = str(checkpoint_path)
        self._run_lock = threading.Lock()
        self._sam3_text_model = None
        self.predictor = self._build_predictor(str(checkpoint_path))

    @classmethod
    def from_predictor(
        cls, variant: str, predictor, device: str | torch.device = "cpu"
    ):
        """Testing/embedding constructor that skips checkpoint loading."""
        instance = cls.__new__(cls)
        instance.variant = variant
        instance.kind = VARIANT_INFO[variant]["kind"]
        instance.device = torch.device(device)
        instance.compile_model = False
        instance.checkpoint_path = None
        instance._run_lock = threading.Lock()
        instance._sam3_text_model = None
        instance.predictor = predictor
        return instance

    def _build_predictor(self, checkpoint_path: str):
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"SAM2Matting checkpoint not found: {checkpoint_path}"
            )
        if self.kind == "sam2":
            return self._build_sam2(checkpoint_path)
        return self._build_sam3(checkpoint_path)

    def _build_sam2(self, checkpoint_path: str):
        # The upstream fork adds matting-specific model classes to SAM2. Keep
        # it under a private top-level name so ComfyUI nodes that import the
        # standard ``sam2`` package can safely coexist in the same process.
        _activate_vendored_package(SAM2_VENDOR_PACKAGE)
        from comfyui_sam2matting_sam2.build_sam import (
            build_sam2matting_video_predictor,
        )

        # Keep upstream's useful mask post-processing but disable its optional
        # 8-pixel CUDA connected-components pass. Custom-node installs do not
        # build the private connected-components extension, and the upstream
        # function otherwise emits a warning before falling back on every run.
        overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=0",
        ]
        if self.compile_model:
            overrides.append("++model.compile_image_encoder=True")
        return build_sam2matting_video_predictor(
            VARIANT_INFO[self.variant]["config"],
            checkpoint_path,
            device=str(self.device),
            hydra_overrides_extra=overrides,
            apply_postprocessing=False,
        )

    def _build_sam3(self, checkpoint_path: str):
        if self.device.type != "cuda":
            raise RuntimeError(
                "The upstream SAM3 video predictor currently requires CUDA."
            )
        _activate_vendored_package(SAM3_VENDOR_PACKAGE)
        from iopath.common.file_io import g_pathmgr
        from comfyui_sam2matting_sam3.model.sam3matting_video_predictor import (
            build_sam3matting_video_predictor,
        )

        with g_pathmgr.open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = torch.load(
                checkpoint_file, map_location="cpu", weights_only=True
            )["model"]
        tracker_state = {}
        for key, value in checkpoint.items():
            if key.startswith("detector.backbone.vision_backbone."):
                tracker_state[key.removeprefix("detector.")] = value
            elif key.startswith("tracker."):
                tracker_state[key.removeprefix("tracker.")] = value

        predictor = build_sam3matting_video_predictor(
            checkpoint=None, device=str(self.device)
        )
        predictor.load_state_dict(tracker_state, strict=False)
        if self.compile_model:
            trunk = predictor.backbone.vision_backbone.trunk
            trunk.forward = torch.compile(
                trunk.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        return predictor

    def _get_sam3_text_model(self):
        if self.kind != "sam3":
            raise ValueError(
                "SAM3 text prompting requires the loader variant to be set to sam3"
            )
        if self.checkpoint_path is None:
            raise RuntimeError("The SAM3 text model has no checkpoint path")
        if self._sam3_text_model is None:
            _activate_vendored_package(SAM3_VENDOR_PACKAGE)
            from comfyui_sam2matting_sam3.model_builder import build_sam3_image_model

            bpe_path = VENDOR_DIR / SAM3_VENDOR_PACKAGE / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.is_file():
                raise FileNotFoundError(
                    f"SAM3 tokenizer vocabulary not found: {bpe_path}"
                )
            self._sam3_text_model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                device="cpu",
                eval_mode=True,
                checkpoint_path=self.checkpoint_path,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
        return self._sam3_text_model

    def text_seed_mask(
        self,
        images: torch.Tensor,
        text_prompt: str,
        frame_index: int = 0,
        confidence_threshold: float = 0.5,
        selection: str = "highest_score",
        interrupt_callback: InterruptCallback | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Ground a text prompt on one frame and return one binary seed mask."""
        if self.kind != "sam3":
            raise ValueError(
                "SAM3 text prompting requires the loader variant to be set to sam3"
            )
        prompt = str(text_prompt).strip()
        if not prompt:
            raise ValueError("text_prompt must not be empty")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if selection not in SAM3_TEXT_SELECTIONS:
            raise ValueError(
                f"Unknown SAM3 text selection {selection!r}; choose one of "
                f"{SAM3_TEXT_SELECTIONS}"
            )

        images = torch.as_tensor(images)
        if images.ndim != 4:
            raise ValueError(
                "images must be one ordered ComfyUI IMAGE batch with shape "
                f"[frames,height,width,channels], got {tuple(images.shape)}"
            )
        frame_count, _height, _width, channels = map(int, images.shape)
        if frame_count < 1:
            raise ValueError("images must contain at least one frame")
        if channels < 3:
            raise ValueError(
                f"images must have at least three color channels, got {channels}"
            )
        if not 0 <= int(frame_index) < frame_count:
            raise ValueError(
                f"frame_index must be between 0 and {frame_count - 1}, got {frame_index}"
            )

        frame_chw = (
            images[int(frame_index), ..., :3]
            .detach()
            .float()
            .cpu()
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        if interrupt_callback is not None:
            interrupt_callback()

        with self._run_lock:
            # The detector and temporal tracker each contain a large vision
            # backbone. Keep only the active one on CUDA so prompt refinement
            # does not permanently double the node's VRAM footprint.
            text_model = None
            processor = None
            state = None
            _move_module_and_tensor_caches(self.predictor, "cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                text_model = self._get_sam3_text_model()
                _move_module_and_tensor_caches(text_model, self.device)
                from comfyui_sam2matting_sam3.model.sam3_image_processor import (
                    Sam3Processor,
                )

                with torch.inference_mode(), _autocast_context(self.device):
                    processor = Sam3Processor(
                        text_model,
                        device=str(self.device),
                        confidence_threshold=float(confidence_threshold),
                    )
                    state = processor.set_image(frame_chw)
                    if interrupt_callback is not None:
                        interrupt_callback()
                    state = processor.set_text_prompt(prompt=prompt, state=state)
                    mask, score = select_sam3_text_mask(
                        state["masks"], state["scores"], selection
                    )
                return mask.unsqueeze(0), score
            finally:
                state = None
                processor = None
                if text_model is not None:
                    _move_module_and_tensor_caches(text_model, "cpu")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                _move_module_and_tensor_caches(self.predictor, self.device)

    def matte_video(
        self,
        images: torch.Tensor,
        initial_mask: torch.Tensor,
        mask_frame: int = 0,
        mask_threshold: float = 0.5,
        memory_mode: str = "balanced",
        progress_callback: ProgressCallback | None = None,
        interrupt_callback: InterruptCallback | None = None,
    ) -> torch.Tensor:
        """Propagate one seed mask through one ordered IMAGE batch."""
        images = torch.as_tensor(images)
        if images.ndim != 4:
            raise ValueError(
                "images must be one ordered ComfyUI IMAGE batch with shape "
                f"[frames,height,width,channels], got {tuple(images.shape)}"
            )
        frame_count, height, width, channels = map(int, images.shape)
        if frame_count < 1:
            raise ValueError("images must contain at least one frame")
        if channels < 3:
            raise ValueError(
                f"images must have at least three color channels, got {channels}"
            )
        if not 0 <= int(mask_frame) < frame_count:
            raise ValueError(
                f"mask_frame must be between 0 and {frame_count - 1}, got {mask_frame}"
            )
        if memory_mode not in MEMORY_MODES:
            raise ValueError(
                f"Unknown memory_mode {memory_mode!r}; choose one of {tuple(MEMORY_MODES)}"
            )

        seed_mask = prepare_seed_mask(
            initial_mask,
            int(mask_frame),
            frame_count,
            height,
            width,
            float(mask_threshold),
        )
        offload_video, offload_state = MEMORY_MODES[memory_mode]
        image_size = int(self.predictor.image_size)

        if interrupt_callback is not None:
            interrupt_callback()
        frames = _prepare_frames(
            images,
            image_size,
            self.kind,
            self.device,
            offload_video,
            interrupt_callback,
        )
        alphas: list[torch.Tensor | None] = [None] * frame_count

        def collect_alpha(frame_index: int, alpha: torch.Tensor) -> None:
            alphas[frame_index] = alpha

        self._propagate_frame_sequence(
            frames=frames,
            frame_count=frame_count,
            height=height,
            width=width,
            seed_mask=seed_mask,
            mask_frame=int(mask_frame),
            offload_video_to_cpu=offload_video,
            offload_state_to_cpu=offload_state,
            alpha_callback=collect_alpha,
            progress_callback=progress_callback,
            interrupt_callback=interrupt_callback,
        )
        return torch.stack(alphas, dim=0).float()  # type: ignore[arg-type]

    def matte_frame_sequence(
        self,
        frames: Sequence[torch.Tensor],
        height: int,
        width: int,
        initial_mask: torch.Tensor,
        mask_frame: int = 0,
        mask_threshold: float = 0.5,
        memory_mode: str = "balanced",
        alpha_callback: AlphaCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        interrupt_callback: InterruptCallback | None = None,
        bounded_state: bool = True,
    ) -> None:
        """Propagate over a normalized, file-backed frame sequence.

        Unlike :meth:`matte_video`, this API never collects a full alpha batch.
        Each completed full-resolution matte is delivered to ``alpha_callback``
        and can be written to disk immediately. By default, temporal outputs
        older than the predictor's attention horizon are discarded as the
        sequence advances, bounding both GPU and host tracking-state memory.
        """
        frame_count = len(frames)
        if frame_count < 1:
            raise ValueError("frames must contain at least one frame")
        if height < 1 or width < 1:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        if not 0 <= int(mask_frame) < frame_count:
            raise ValueError(
                f"mask_frame must be between 0 and {frame_count - 1}, got {mask_frame}"
            )
        if memory_mode not in MEMORY_MODES:
            raise ValueError(
                f"Unknown memory_mode {memory_mode!r}; choose one of {tuple(MEMORY_MODES)}"
            )
        if alpha_callback is None:
            raise ValueError("alpha_callback is required for file-backed propagation")

        seed_mask = prepare_seed_mask(
            initial_mask,
            int(mask_frame),
            frame_count,
            int(height),
            int(width),
            float(mask_threshold),
        )
        _ignored_video_mode, offload_state = MEMORY_MODES[memory_mode]
        if interrupt_callback is not None:
            interrupt_callback()
        self._propagate_frame_sequence(
            frames=frames,
            frame_count=frame_count,
            height=int(height),
            width=int(width),
            seed_mask=seed_mask,
            mask_frame=int(mask_frame),
            offload_video_to_cpu=True,
            offload_state_to_cpu=offload_state,
            alpha_callback=alpha_callback,
            progress_callback=progress_callback,
            interrupt_callback=interrupt_callback,
            bounded_state=bool(bounded_state),
        )

    def _propagate_frame_sequence(
        self,
        *,
        frames,
        frame_count: int,
        height: int,
        width: int,
        seed_mask: torch.Tensor,
        mask_frame: int,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        alpha_callback: AlphaCallback,
        progress_callback: ProgressCallback | None,
        interrupt_callback: InterruptCallback | None,
        bounded_state: bool = False,
    ) -> None:
        """Run one predictor state and emit alphas without retaining them."""
        with self._run_lock, torch.inference_mode(), _autocast_context(self.device):
            if self.kind == "sam2":
                state = _init_sam2_state(
                    self.predictor,
                    frames,
                    height,
                    width,
                    offload_video_to_cpu,
                    offload_state_to_cpu,
                )
            else:
                state = _init_sam3_state(
                    self.predictor,
                    frames,
                    height,
                    width,
                    offload_video_to_cpu,
                    offload_state_to_cpu,
                )

            completed: set[int] = set()
            try:
                if interrupt_callback is not None:
                    interrupt_callback()
                self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=mask_frame,
                    obj_id=1,
                    mask=seed_mask.to(self.device),
                )
                for reverse, result in _propagation_passes(
                    self.predictor, state, mask_frame
                ):
                    if interrupt_callback is not None:
                        interrupt_callback()
                    frame_index, raw_alpha = _extract_propagated_alpha(result)
                    if not 0 <= frame_index < frame_count:
                        raise RuntimeError(
                            f"Predictor returned out-of-range frame index {frame_index}"
                        )
                    alpha = _alpha_to_2d(raw_alpha, height, width)
                    if bounded_state:
                        _prune_temporal_state(
                            self.predictor,
                            state,
                            current_frame=frame_index,
                            mask_frame=mask_frame,
                            reverse=reverse,
                        )
                    alpha_callback(frame_index, alpha)
                    completed.add(frame_index)
                    if progress_callback is not None:
                        progress_callback(len(completed), frame_count)

                missing = [
                    index for index in range(frame_count) if index not in completed
                ]
                if missing:
                    raise RuntimeError(
                        "Temporal propagation did not return every input frame; missing "
                        + ", ".join(map(str, missing[:20]))
                    )
            finally:
                try:
                    self.predictor.reset_state(state)
                finally:
                    state.clear()


def make_checkerboard_preview(
    images: torch.Tensor,
    alpha: torch.Tensor,
    cell_size: int = 24,
) -> torch.Tensor:
    """Composite an RGB batch over a neutral checkerboard for visual checking."""
    rgb = torch.as_tensor(images).detach().float().cpu()[..., :3].clamp(0.0, 1.0)
    alpha = torch.as_tensor(alpha).detach().float().cpu().clamp(0.0, 1.0)
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    yy = torch.arange(height)[:, None] // cell_size
    xx = torch.arange(width)[None, :] // cell_size
    pattern = ((yy + xx) % 2).float()
    checker = (0.28 + pattern * 0.18)[None, ..., None].expand_as(rgb)
    return rgb * alpha[..., None] + checker * (1.0 - alpha[..., None])
