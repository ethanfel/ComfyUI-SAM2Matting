"""Disk-backed native-video processing for bounded host-memory use."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import av
import numpy as np
import torch
from PIL import Image

try:
    from .video_model import frame_normalization
except ImportError:  # Standalone import used by the checkpoint-free test suite.
    from video_model import frame_normalization


ProgressCallback = Callable[[int, int], None]
InterruptCallback = Callable[[], None]


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


class DiskFrameSequence:
    """Random-access normalized tracking frames stored as indexed JPEGs."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        frame_count: int,
        image_size: int,
        kind: str,
    ) -> None:
        self.path = Path(path)
        self.frame_count = int(frame_count)
        self.image_size = int(image_size)
        mean, std = frame_normalization(kind)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self._file = self.path.open("rb")
        self._records: list[tuple[int, int]] = []
        for _index in range(self.frame_count):
            length_bytes = self._file.read(8)
            if len(length_bytes) != 8:
                raise IOError("The tracking-frame cache is incomplete")
            length = int.from_bytes(length_bytes, "little")
            offset = self._file.tell()
            self._records.append((offset, length))
            self._file.seek(length, os.SEEK_CUR)

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += self.frame_count
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        offset, length = self._records[index]
        self._file.seek(offset)
        payload = self._file.read(length)
        if len(payload) != length:
            raise IOError(f"Tracking frame {index} is incomplete")
        with Image.open(io.BytesIO(payload)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        frame = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        return (frame - self.mean) / self.std

    def close(self) -> None:
        self._file.close()


class DiskAlphaStore:
    """Random-access, lossless 8-bit PNG mattes in one indexed disk file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        frame_count: int,
        height: int,
        width: int,
    ) -> None:
        self.path = Path(path)
        self.frame_count = int(frame_count)
        self.height = int(height)
        self.width = int(width)
        self._file = self.path.open("w+b")
        self._records: list[tuple[int, int] | None] = [None] * self.frame_count

    def write(self, frame_index: int, alpha: torch.Tensor) -> None:
        matte = torch.as_tensor(alpha).detach().float().cpu()
        if tuple(matte.shape) != (self.height, self.width):
            raise ValueError(
                f"Alpha {frame_index} has shape {tuple(matte.shape)}, expected "
                f"{(self.height, self.width)}"
            )
        array = matte.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        payload = io.BytesIO()
        Image.fromarray(array).save(payload, format="PNG", compress_level=4)
        encoded = payload.getvalue()
        self._file.seek(0, os.SEEK_END)
        offset = self._file.tell()
        self._file.write(encoded)
        self._records[frame_index] = (offset, len(encoded))

    def as_float(self, frame_index: int) -> np.ndarray:
        record = self._records[frame_index]
        if record is None:
            raise RuntimeError(f"Alpha frame {frame_index} was not written")
        offset, length = record
        self._file.flush()
        self._file.seek(offset)
        payload = self._file.read(length)
        if len(payload) != length:
            raise IOError(f"Alpha frame {frame_index} is incomplete")
        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image, dtype=np.float32) / 255.0

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self.flush()
            self._file.close()


def prepare_tracking_frames(
    video,
    path: str | os.PathLike[str],
    image_size: int,
    kind: str,
    progress_callback: ProgressCallback | None = None,
    interrupt_callback: InterruptCallback | None = None,
) -> tuple[DiskFrameSequence, VideoInfo]:
    """Decode once and write compact model-resolution frames to indexed JPEGs."""
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
    with path.open("wb") as output:
        for rgb in iter_rgb_frames(video):
            if interrupt_callback is not None:
                interrupt_callback()
            current_height, current_width = map(int, rgb.shape[:2])
            if frame_count == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise ValueError(
                    "Video resolution changes mid-stream "
                    f"({width}x{height} -> {current_width}x{current_height})"
                )
            tracking_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            tracking_frame = tracking_frame.reformat(
                width=int(image_size),
                height=int(image_size),
                format="rgb24",
            )
            payload = io.BytesIO()
            Image.fromarray(tracking_frame.to_ndarray()).save(
                payload,
                format="JPEG",
                quality=95,
                subsampling=0,
            )
            encoded = payload.getvalue()
            output.write(len(encoded).to_bytes(8, "little"))
            output.write(encoded)
            frame_count += 1
            if progress_callback is not None:
                progress_callback(frame_count, max(estimated, frame_count))

    if frame_count == 0:
        raise ValueError("The VIDEO input contains no decodable frames")
    sequence = DiskFrameSequence(path, frame_count, image_size, kind)
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


def _encode_video_only(
    video,
    alpha_store: DiskAlphaStore,
    info: VideoInfo,
    background: tuple[int, int, int],
    output_path: str | os.PathLike[str],
    output_fps: float,
    crf: int,
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
        stream = output.add_stream("h264", rate=output_rate)
        stream.width = info.width
        stream.height = info.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf))}
        stream.codec_context.max_b_frames = 0
        time_base = Fraction(output_rate.denominator, output_rate.numerator)
        encoded = 0
        decoded = 0
        for frame_index, rgb in enumerate(iter_rgb_frames(video)):
            if frame_index >= info.frame_count:
                break
            decoded += 1
            if interrupt_callback is not None:
                interrupt_callback()
            if tuple(rgb.shape[:2]) != (info.height, info.width):
                raise ValueError("Decoded video dimensions changed between passes")
            source_for_output = min(
                info.frame_count - 1,
                int(Fraction(encoded) * info.frame_rate / output_rate),
            )
            if source_for_output != frame_index:
                if progress_callback is not None:
                    progress_callback(decoded, info.frame_count)
                continue
            composited = _composite_rgb(
                rgb,
                alpha_store.as_float(frame_index),
                background,
            )
            while encoded < output_frame_count:
                source_for_output = min(
                    info.frame_count - 1,
                    int(Fraction(encoded) * info.frame_rate / output_rate),
                )
                if source_for_output != frame_index:
                    break
                frame = av.VideoFrame.from_ndarray(composited, format="rgb24")
                frame.pts = encoded
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    output.mux(packet)
                encoded += 1
            if progress_callback is not None:
                progress_callback(decoded, info.frame_count)
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
    progress_callback: ProgressCallback | None = None,
    interrupt_callback: InterruptCallback | None = None,
) -> None:
    """Composite and encode without constructing an IMAGE batch."""
    output_info = _encode_video_only(
        video,
        alpha_store,
        info,
        background,
        video_only_path,
        output_fps,
        crf,
        progress_callback,
        interrupt_callback,
    )
    if preserve_audio and _mux_source_audio(
        video, video_only_path, output_path, output_info
    ):
        return
    os.replace(video_only_path, output_path)
