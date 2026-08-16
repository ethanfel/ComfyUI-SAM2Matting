"""Disk-backed native-video processing for bounded host-memory use."""

from __future__ import annotations

import io
import logging
import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import av
import numpy as np
import torch
import zstandard
from PIL import Image

try:
    from .video_model import frame_normalization
except ImportError:  # Standalone import used by the checkpoint-free test suite.
    from video_model import frame_normalization


ProgressCallback = Callable[[int, int], None]
InterruptCallback = Callable[[], None]
TRACKING_CACHE_MODES = ("lossless_zstd", "jpeg_low_disk", "raw_fast")
VIDEO_ENCODERS = ("auto", "h264_nvenc", "libx264")


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    width: int
    height: int
    frame_rate: Fraction


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse a six-digit RGB color accepted by the ComfyUI widget."""
    color = str(value).strip()
    if color.startswith("#"):
        color = color[1:]
    if len(color) != 6:
        raise ValueError(
            f"background_color must be a six-digit RGB hex color, got {value!r}"
        )
    try:
        return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(
            f"background_color must be a six-digit RGB hex color, got {value!r}"
        ) from exc


def _source(video):
    source = video.get_stream_source()
    if isinstance(source, io.BytesIO):
        source.seek(0)
    return source


def _active_window(video) -> tuple[float, float]:
    getter = getattr(video, "get_active_trim_window", None)
    if getter is None:
        return 0.0, 0.0
    start_time, duration = getter()
    return max(0.0, float(start_time)), max(0.0, float(duration))


def _rotate_rgb(frame: av.VideoFrame, rgb: np.ndarray) -> np.ndarray:
    rotation = int(round(float(getattr(frame, "rotation", 0) or 0) / 90.0)) % 4
    if rotation:
        rgb = np.rot90(rgb, k=rotation, axes=(0, 1)).copy()
    return rgb


def iter_rgb_frames(video) -> Iterator[np.ndarray]:
    """Decode the active native-video window one RGB frame at a time."""
    start_time, duration = _active_window(video)
    end_time = start_time + duration if duration else None
    with av.open(_source(video), mode="r") as container:
        if not container.streams.video:
            raise ValueError("The VIDEO input contains no video stream")
        stream = container.streams.video[0]
        if start_time and stream.time_base:
            container.seek(
                int(start_time / stream.time_base),
                stream=stream,
                backward=True,
            )
        fallback_rate = Fraction(stream.average_rate or 1)
        fallback_index = 0
        for frame in container.decode(stream):
            if frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
            else:
                timestamp = fallback_index / float(fallback_rate)
            fallback_index += 1
            if timestamp + 1e-9 < start_time:
                continue
            if end_time is not None and timestamp >= end_time - 1e-9:
                break
            yield _rotate_rgb(frame, frame.to_ndarray(format="rgb24"))


def _encode_tracking_frame(
    rgb: np.ndarray,
    image_size: int,
    cache_mode: str,
) -> bytes:
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
        width=int(image_size),
        height=int(image_size),
        format="rgb24",
    )
    array = np.ascontiguousarray(frame.to_ndarray(), dtype=np.uint8)
    if cache_mode == "raw_fast":
        return array.tobytes()
    if cache_mode == "lossless_zstd":
        return zstandard.ZstdCompressor(level=1).compress(array.tobytes())
    if cache_mode == "jpeg_low_disk":
        payload = io.BytesIO()
        Image.fromarray(array).save(
            payload,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        return payload.getvalue()
    raise ValueError(
        f"Unknown tracking cache mode {cache_mode!r}; choose one of "
        f"{TRACKING_CACHE_MODES}"
    )


class DiskFrameSequence:
    """Random-access normalized tracking frames in one indexed disk file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        frame_count: int,
        image_size: int,
        kind: str,
        cache_mode: str = "lossless_zstd",
        prefetch_frames: int = 0,
        worker_threads: int = 1,
    ) -> None:
        if cache_mode not in TRACKING_CACHE_MODES:
            raise ValueError(
                f"Unknown tracking cache mode {cache_mode!r}; choose one of "
                f"{TRACKING_CACHE_MODES}"
            )
        self.path = Path(path)
        self.frame_count = int(frame_count)
        self.image_size = int(image_size)
        self.cache_mode = cache_mode
        self.prefetch_frames = max(0, int(prefetch_frames))
        mean, std = frame_normalization(kind)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self._records: list[tuple[int, int]] = []
        with self.path.open("rb") as cache:
            for _index in range(self.frame_count):
                length_bytes = cache.read(8)
                if len(length_bytes) != 8:
                    raise IOError("The tracking-frame cache is incomplete")
                length = int.from_bytes(length_bytes, "little")
                offset = cache.tell()
                self._records.append((offset, length))
                cache.seek(length, os.SEEK_CUR)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max(1, min(int(worker_threads), self.prefetch_frames)),
                thread_name_prefix="sam2-frame-prefetch",
            )
            if self.prefetch_frames > 0
            else None
        )
        self._prefetched: dict[int, Future[torch.Tensor]] = {}
        self._prefetch_direction = 0

    def __len__(self) -> int:
        return self.frame_count

    def _decode(self, index: int) -> torch.Tensor:
        offset, length = self._records[index]
        with self.path.open("rb") as cache:
            cache.seek(offset)
            payload = cache.read(length)
        if len(payload) != length:
            raise IOError(f"Tracking frame {index} is incomplete")
        if self.cache_mode == "jpeg_low_disk":
            with Image.open(io.BytesIO(payload)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        else:
            if self.cache_mode == "lossless_zstd":
                payload = zstandard.ZstdDecompressor().decompress(
                    payload,
                    max_output_size=self.image_size * self.image_size * 3,
                )
            array = np.frombuffer(payload, dtype=np.uint8)
            expected = self.image_size * self.image_size * 3
            if array.size != expected:
                raise IOError(
                    f"Tracking frame {index} has {array.size} bytes, expected "
                    f"{expected}"
                )
            array = array.reshape(self.image_size, self.image_size, 3).copy()
        frame = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        return (frame - self.mean) / self.std

    def set_prefetch_direction(self, start_index: int, direction: int) -> None:
        """Prime a bounded read/decode window for a propagation pass."""
        if self._executor is None:
            return
        self._prefetch_direction = 1 if int(direction) >= 0 else -1
        for index, future in list(self._prefetched.items()):
            if not future.done():
                future.cancel()
            self._prefetched.pop(index, None)
        self._schedule_after(int(start_index) - self._prefetch_direction)

    def _schedule_after(self, index: int) -> None:
        if self._executor is None or self._prefetch_direction == 0:
            return
        for distance in range(1, self.prefetch_frames + 1):
            candidate = index + distance * self._prefetch_direction
            if not 0 <= candidate < self.frame_count:
                break
            if candidate not in self._prefetched:
                self._prefetched[candidate] = self._executor.submit(
                    self._decode, candidate
                )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += self.frame_count
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        future = self._prefetched.pop(index, None)
        frame = future.result() if future is not None else self._decode(index)
        self._schedule_after(index)
        return frame

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._prefetched.clear()


class DiskAlphaStore:
    """Random-access lossless mattes with optional bounded async PNG writes."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        frame_count: int,
        height: int,
        width: int,
        worker_threads: int = 1,
        queue_depth: int = 1,
    ) -> None:
        self.path = Path(path)
        self.frame_count = int(frame_count)
        self.height = int(height)
        self.width = int(width)
        self._file = self.path.open("w+b")
        self._records: list[tuple[int, int] | None] = [None] * self.frame_count
        self._write_lock = threading.Lock()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max(1, int(worker_threads)),
                thread_name_prefix="sam2-alpha-write",
            )
            if int(worker_threads) > 1
            else None
        )
        self._slots = threading.Semaphore(max(1, int(queue_depth)))
        self._futures: list[Future[None]] = []

    @staticmethod
    def _png_bytes(array: np.ndarray) -> bytes:
        payload = io.BytesIO()
        Image.fromarray(array).save(payload, format="PNG", compress_level=4)
        return payload.getvalue()

    def _encode_and_append(self, frame_index: int, array: np.ndarray) -> None:
        encoded = self._png_bytes(array)
        with self._write_lock:
            self._file.seek(0, os.SEEK_END)
            offset = self._file.tell()
            self._file.write(encoded)
            self._records[frame_index] = (offset, len(encoded))

    def write(self, frame_index: int, alpha: torch.Tensor) -> None:
        matte = torch.as_tensor(alpha).detach().float().cpu()
        if tuple(matte.shape) != (self.height, self.width):
            raise ValueError(
                f"Alpha {frame_index} has shape {tuple(matte.shape)}, expected "
                f"{(self.height, self.width)}"
            )
        array = matte.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        if self._executor is None:
            self._encode_and_append(frame_index, array)
            return
        self._slots.acquire()
        future = self._executor.submit(self._encode_and_append, frame_index, array)
        future.add_done_callback(lambda _future: self._slots.release())
        self._futures.append(future)

    def as_float(self, frame_index: int) -> np.ndarray:
        return self.as_uint8(frame_index).astype(np.float32) / 255.0

    def as_uint8(self, frame_index: int) -> np.ndarray:
        """Read one matte without expanding it to full-resolution float32."""
        record = self._records[frame_index]
        if record is None:
            raise RuntimeError(f"Alpha frame {frame_index} was not written")
        offset, length = record
        with self.path.open("rb") as cache:
            cache.seek(offset)
            payload = cache.read(length)
        if len(payload) != length:
            raise IOError(f"Alpha frame {frame_index} is incomplete")
        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image, dtype=np.uint8).copy()

    def flush(self) -> None:
        futures, self._futures = self._futures, []
        for future in futures:
            future.result()
        with self._write_lock:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self.flush()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self._file.close()


def prepare_tracking_frames(
    video,
    path: str | os.PathLike[str],
    image_size: int,
    kind: str,
    cache_mode: str = "lossless_zstd",
    worker_threads: int = 4,
    pipeline_depth: int = 8,
    progress_callback: ProgressCallback | None = None,
    interrupt_callback: InterruptCallback | None = None,
) -> tuple[DiskFrameSequence, VideoInfo]:
    """Decode once and write model-resolution frames to a bounded disk cache."""
    if cache_mode not in TRACKING_CACHE_MODES:
        raise ValueError(
            f"Unknown tracking cache mode {cache_mode!r}; choose one of "
            f"{TRACKING_CACHE_MODES}"
        )
    worker_threads = max(1, int(worker_threads))
    pipeline_depth = max(1, int(pipeline_depth))
    frame_rate = Fraction(video.get_frame_rate())
    if frame_rate <= 0:
        frame_rate = Fraction(1)
    try:
        estimated = max(1, int(round(float(video.get_duration()) * frame_rate)))
    except (AttributeError, TypeError, ValueError):
        estimated = 1

    frame_count = 0
    width = height = 0
    path = Path(path)
    pending: deque[Future[bytes]] = deque()

    def write_next(output) -> None:
        nonlocal frame_count
        encoded = pending.popleft().result()
        output.write(len(encoded).to_bytes(8, "little"))
        output.write(encoded)
        frame_count += 1
        if progress_callback is not None:
            progress_callback(frame_count, max(estimated, frame_count))

    with (
        path.open("wb") as output,
        ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix="sam2-frame-prepare",
        ) as executor,
    ):
        for rgb in iter_rgb_frames(video):
            if interrupt_callback is not None:
                interrupt_callback()
            current_height, current_width = map(int, rgb.shape[:2])
            if frame_count == 0 and not pending:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise ValueError(
                    "Video resolution changes mid-stream "
                    f"({width}x{height} -> {current_width}x{current_height})"
                )
            pending.append(
                executor.submit(
                    _encode_tracking_frame,
                    rgb,
                    int(image_size),
                    cache_mode,
                )
            )
            if len(pending) >= pipeline_depth:
                write_next(output)
        while pending:
            write_next(output)

    if frame_count == 0:
        raise ValueError("The VIDEO input contains no decodable frames")
    sequence = DiskFrameSequence(
        path,
        frame_count,
        image_size,
        kind,
        cache_mode=cache_mode,
        prefetch_frames=pipeline_depth,
        worker_threads=worker_threads,
    )
    return sequence, VideoInfo(frame_count, width, height, frame_rate)


def _composite_rgb(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background: tuple[int, int, int],
) -> np.ndarray:
    """Composite one uint8 RGB frame over a solid color."""
    foreground = np.asarray(rgb, dtype=np.float32)
    opacity = np.asarray(alpha, dtype=np.float32)[..., None]
    foreground *= opacity
    foreground += np.asarray(background, dtype=np.float32) * (1.0 - opacity)
    return np.clip(np.rint(foreground), 0, 255).astype(np.uint8)


def _temporally_stabilized_alpha(
    alpha_store: DiskAlphaStore,
    frame_index: int,
    frame_count: int,
    strength: float,
) -> np.ndarray:
    """Blend one matte toward its three-frame temporal median.

    For monotonic boundary motion, the current frame is already the median and
    remains unchanged. A one-frame inward or outward excursion is the outlier,
    so it is pulled toward the two neighboring mattes. Only three uint8 mattes
    are read, keeping this independent of clip length.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    current = alpha_store.as_uint8(frame_index)
    if strength <= 0.0 or frame_index == 0 or frame_index == frame_count - 1:
        return current.astype(np.float32) / 255.0

    previous = alpha_store.as_uint8(frame_index - 1)
    following = alpha_store.as_uint8(frame_index + 1)
    lower = np.minimum(previous, following)
    upper = np.maximum(previous, following)
    median = np.clip(current, lower, upper)

    stabilized = current.astype(np.float32)
    stabilized *= np.float32(1.0 - strength)
    stabilized += median.astype(np.float32) * np.float32(strength)
    stabilized *= np.float32(1.0 / 255.0)
    return stabilized


def _encode_video_only(
    video,
    alpha_store: DiskAlphaStore,
    info: VideoInfo,
    background: tuple[int, int, int],
    output_path: str | os.PathLike[str],
    output_fps: float,
    crf: int,
    encoder: str,
    worker_threads: int,
    pipeline_depth: int,
    edge_stabilization: float,
    progress_callback: ProgressCallback | None,
    interrupt_callback: InterruptCallback | None,
) -> VideoInfo:
    if info.width % 2 or info.height % 2:
        raise ValueError(
            "H.264 output requires even video dimensions, got "
            f"{info.width}x{info.height}"
        )
    output_rate = (
        info.frame_rate
        if float(output_fps) <= 0.0
        else Fraction(str(float(output_fps))).limit_denominator(100_000)
    )
    output_frame_count = max(
        1,
        int(round(Fraction(info.frame_count) * output_rate / info.frame_rate)),
    )
    with av.open(
        str(output_path),
        mode="w",
        format="mp4",
        options={"movflags": "use_metadata_tags+faststart"},
    ) as output:
        stream = output.add_stream(encoder, rate=output_rate)
        stream.width = info.width
        stream.height = info.height
        stream.pix_fmt = "yuv420p"
        stream.options = (
            {"cq": str(int(crf)), "preset": "p4"}
            if encoder == "h264_nvenc"
            else {"crf": str(int(crf))}
        )
        stream.codec_context.max_b_frames = 0
        time_base = Fraction(output_rate.denominator, output_rate.numerator)
        encoded = 0
        decoded = 0
        scheduled = 0
        pending: deque[tuple[int, int, Future[np.ndarray]]] = deque()

        def composite_frame(frame_index: int, rgb: np.ndarray) -> np.ndarray:
            return _composite_rgb(
                rgb,
                _temporally_stabilized_alpha(
                    alpha_store,
                    frame_index,
                    info.frame_count,
                    edge_stabilization,
                ),
                background,
            )

        def encode_next() -> None:
            nonlocal encoded
            _frame_index, repeat_count, future = pending.popleft()
            composited = future.result()
            for _repeat in range(repeat_count):
                frame = av.VideoFrame.from_ndarray(composited, format="rgb24")
                frame.pts = encoded
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    output.mux(packet)
                encoded += 1

        worker_threads = max(1, int(worker_threads))
        pipeline_depth = max(1, int(pipeline_depth))
        with ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix="sam2-composite",
        ) as executor:
            for frame_index, rgb in enumerate(iter_rgb_frames(video)):
                if frame_index >= info.frame_count:
                    break
                decoded += 1
                if interrupt_callback is not None:
                    interrupt_callback()
                if tuple(rgb.shape[:2]) != (info.height, info.width):
                    raise ValueError("Decoded video dimensions changed between passes")
                repeat_count = 0
                while scheduled < output_frame_count:
                    source_for_output = min(
                        info.frame_count - 1,
                        int(Fraction(scheduled) * info.frame_rate / output_rate),
                    )
                    if source_for_output != frame_index:
                        break
                    repeat_count += 1
                    scheduled += 1
                if repeat_count:
                    pending.append(
                        (
                            frame_index,
                            repeat_count,
                            executor.submit(composite_frame, frame_index, rgb),
                        )
                    )
                    if len(pending) >= pipeline_depth:
                        encode_next()
                if progress_callback is not None:
                    progress_callback(decoded, info.frame_count)
            while pending:
                encode_next()
        if decoded != info.frame_count:
            raise RuntimeError(
                f"Video decode returned {decoded} frames on the output pass; "
                f"tracking used {info.frame_count}"
            )
        if encoded != output_frame_count:
            raise RuntimeError(
                f"FPS conversion produced {encoded} frames, expected "
                f"{output_frame_count}"
            )
        for packet in stream.encode(None):
            output.mux(packet)
    return VideoInfo(output_frame_count, info.width, info.height, output_rate)


def _audio_stream(container):
    for stream in reversed(container.streams.audio):
        if stream.codec_context is not None:
            return stream
    return None


def _encoded_audio_packets(
    container,
    source_stream,
    output_stream,
    start_time: float,
    max_samples: int,
):
    sample_rate = int(output_stream.rate)
    layout = output_stream.layout.name
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
    written = 0
    started = False
    if start_time:
        container.seek(int(start_time * av.time_base), backward=True)

    for packet in container.demux(source_stream):
        try:
            decoded_frames = packet.decode()
        except av.error.FFmpegError:
            continue
        for decoded in decoded_frames:
            for frame in resampler.resample(decoded):
                array = frame.to_ndarray()
                frame_time = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None
                    else None
                )
                skip = 0
                if not started:
                    if frame_time is not None:
                        skip = max(
                            0, int(round((start_time - frame_time) * sample_rate))
                        )
                    if skip >= frame.samples:
                        continue
                    started = True
                remaining = max_samples - written
                if remaining <= 0:
                    break
                array = np.ascontiguousarray(array[..., skip : skip + remaining])
                if array.shape[-1] == 0:
                    continue
                output_frame = av.AudioFrame.from_ndarray(
                    array,
                    format="fltp",
                    layout=layout,
                )
                output_frame.sample_rate = sample_rate
                output_frame.pts = written
                output_frame.time_base = Fraction(1, sample_rate)
                written += output_frame.samples
                yield from output_stream.encode(output_frame)
            if written >= max_samples:
                break
        if written >= max_samples:
            break
    yield from output_stream.encode(None)


def _packet_time(packet) -> float:
    timestamp = packet.dts if packet.dts is not None else packet.pts
    if timestamp is None or packet.time_base is None:
        return float("inf")
    return float(timestamp * packet.time_base)


def _mux_source_audio(
    video,
    video_only_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    info: VideoInfo,
) -> bool:
    """Copy encoded video and stream-transcode the active source audio to AAC."""
    with (
        av.open(_source(video), mode="r") as source,
        av.open(str(video_only_path), mode="r") as encoded,
    ):
        source_audio = _audio_stream(source)
        if source_audio is None:
            return False
        sample_rate = int(source_audio.codec_context.sample_rate or 0)
        channels = int(source_audio.codec_context.channels or 0)
        if sample_rate <= 0 or channels <= 0:
            logging.warning(
                "Could not determine source audio parameters; omitting audio"
            )
            return False
        layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
        start_time, _duration = _active_window(video)
        max_samples = int(
            round(info.frame_count / float(info.frame_rate) * sample_rate)
        )

        with av.open(
            str(output_path),
            mode="w",
            format="mp4",
            options={"movflags": "use_metadata_tags+faststart"},
        ) as output:
            input_video = encoded.streams.video[0]
            output_video = output.add_stream_from_template(input_video)
            output_audio = output.add_stream("aac", rate=sample_rate, layout=layout)

            def video_packets():
                for packet in encoded.demux(input_video):
                    if packet.dts is None:
                        continue
                    packet.stream = output_video
                    yield packet

            video_iterator = iter(video_packets())
            audio_iterator = iter(
                _encoded_audio_packets(
                    source,
                    source_audio,
                    output_audio,
                    start_time,
                    max_samples,
                )
            )
            video_packet = next(video_iterator, None)
            audio_packet = next(audio_iterator, None)
            while video_packet is not None or audio_packet is not None:
                if audio_packet is None or (
                    video_packet is not None
                    and _packet_time(video_packet) <= _packet_time(audio_packet)
                ):
                    output.mux(video_packet)
                    video_packet = next(video_iterator, None)
                else:
                    output.mux(audio_packet)
                    audio_packet = next(audio_iterator, None)
    return True


def encode_background_video(
    video,
    alpha_store: DiskAlphaStore,
    info: VideoInfo,
    background: tuple[int, int, int],
    output_path: str | os.PathLike[str],
    video_only_path: str | os.PathLike[str],
    output_fps: float = 0.0,
    crf: int = 18,
    preserve_audio: bool = True,
    encoder: str = "auto",
    worker_threads: int = 4,
    pipeline_depth: int = 8,
    edge_stabilization: float = 0.0,
    progress_callback: ProgressCallback | None = None,
    interrupt_callback: InterruptCallback | None = None,
) -> str:
    """Composite and encode without constructing an IMAGE batch."""
    if encoder not in VIDEO_ENCODERS:
        raise ValueError(
            f"Unknown video encoder {encoder!r}; choose one of {VIDEO_ENCODERS}"
        )
    candidates = [encoder]
    if encoder == "auto":
        try:
            av.codec.Codec("h264_nvenc", "w")
        except av.error.FFmpegError:
            candidates = ["libx264"]
        else:
            candidates = ["h264_nvenc", "libx264"]

    output_info = None
    actual_encoder = candidates[0]
    for candidate in candidates:
        Path(video_only_path).unlink(missing_ok=True)
        try:
            output_info = _encode_video_only(
                video,
                alpha_store,
                info,
                background,
                video_only_path,
                output_fps,
                crf,
                candidate,
                worker_threads,
                pipeline_depth,
                edge_stabilization,
                progress_callback,
                interrupt_callback,
            )
            actual_encoder = candidate
            break
        except av.error.FFmpegError:
            if encoder != "auto" or candidate == candidates[-1]:
                raise
            logging.getLogger("SAM2Matting").warning(
                "[stream] NVENC could not start; retrying Stage 3 with libx264"
            )
    if output_info is None:  # Defensive: candidates is always non-empty.
        raise RuntimeError("No usable H.264 video encoder was found")
    if preserve_audio and _mux_source_audio(
        video, video_only_path, output_path, output_info
    ):
        return actual_encoder
    os.replace(video_only_path, output_path)
    return actual_encoder
