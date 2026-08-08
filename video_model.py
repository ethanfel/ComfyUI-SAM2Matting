"""Runtime adapter for temporally propagated SAM2Matting video inference.

This module deliberately has no ComfyUI imports.  It turns a ComfyUI-style
``[frames, height, width, channels]`` tensor into the in-memory frame sequence
expected by the vendored SAM2Matting predictors and normalizes the differing
SAM2/SAM3 propagation outputs into one ordered alpha batch.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F


PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "vendor"
SAM2_VENDOR_PACKAGE = "comfyui_sam2matting_sam2"

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

ProgressCallback = Callable[[int, int], None]
InterruptCallback = Callable[[], None]


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
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


def _device_from_predictor(predictor, fallback: torch.device) -> torch.device:
    value = getattr(predictor, "device", fallback)
    return value if isinstance(value, torch.device) else torch.device(value)


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
    if kind == "sam2":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

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
        "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
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
    alpha_tensor = torch.as_tensor(np.asarray(alpha) if isinstance(alpha, np.ndarray) else alpha)
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
) -> Iterator[tuple]:
    yield from predictor.propagate_in_video(
        state,
        start_frame_idx=mask_frame,
        max_frame_num_to_track=state["num_frames"],
        reverse=False,
        tqdm_disable=True,
    )
    if mask_frame > 0:
        yield from predictor.propagate_in_video(
            state,
            start_frame_idx=mask_frame,
            max_frame_num_to_track=state["num_frames"],
            reverse=True,
            tqdm_disable=True,
        )


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
        self._run_lock = threading.Lock()
        self.predictor = self._build_predictor(str(checkpoint_path))

    @classmethod
    def from_predictor(cls, variant: str, predictor, device: str | torch.device = "cpu"):
        """Testing/embedding constructor that skips checkpoint loading."""
        instance = cls.__new__(cls)
        instance.variant = variant
        instance.kind = VARIANT_INFO[variant]["kind"]
        instance.device = torch.device(device)
        instance.compile_model = False
        instance._run_lock = threading.Lock()
        instance.predictor = predictor
        return instance

    def _build_predictor(self, checkpoint_path: str):
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"SAM2Matting checkpoint not found: {checkpoint_path}")
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
            raise RuntimeError("The upstream SAM3 video predictor currently requires CUDA.")
        _activate_vendored_package("sam3")
        from iopath.common.file_io import g_pathmgr
        from sam3.model.sam3matting_video_predictor import (
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
            raise ValueError(f"images must have at least three color channels, got {channels}")
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
        with self._run_lock, torch.inference_mode(), _autocast_context(self.device):
            frames = _prepare_frames(
                images,
                image_size,
                self.kind,
                self.device,
                offload_video,
                interrupt_callback,
            )
            if self.kind == "sam2":
                state = _init_sam2_state(
                    self.predictor,
                    frames,
                    height,
                    width,
                    offload_video,
                    offload_state,
                )
            else:
                state = _init_sam3_state(
                    self.predictor,
                    frames,
                    height,
                    width,
                    offload_video,
                    offload_state,
                )

            alphas: list[torch.Tensor | None] = [None] * frame_count
            completed: set[int] = set()
            try:
                if interrupt_callback is not None:
                    interrupt_callback()
                self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(mask_frame),
                    obj_id=1,
                    mask=seed_mask.to(self.device),
                )
                for result in _propagation_passes(
                    self.predictor, state, int(mask_frame)
                ):
                    if interrupt_callback is not None:
                        interrupt_callback()
                    frame_index, alpha = _extract_propagated_alpha(result)
                    if not 0 <= frame_index < frame_count:
                        raise RuntimeError(
                            f"Predictor returned out-of-range frame index {frame_index}"
                        )
                    alphas[frame_index] = _alpha_to_2d(alpha, height, width)
                    completed.add(frame_index)
                    if progress_callback is not None:
                        progress_callback(len(completed), frame_count)

                missing = [index for index, alpha in enumerate(alphas) if alpha is None]
                if missing:
                    raise RuntimeError(
                        "Temporal propagation did not return every input frame; missing "
                        + ", ".join(map(str, missing[:20]))
                    )
                return torch.stack(alphas, dim=0).float()  # type: ignore[arg-type]
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
