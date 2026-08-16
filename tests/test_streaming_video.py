from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from streaming_video import (
    TRACKING_CACHE_MODES,
    DiskAlphaStore,
    _temporally_stabilized_alpha,
    encode_background_video,
    parse_hex_color,
    prepare_tracking_frames,
)


class FakeNativeVideo:
    def __init__(self, path, frame_rate=Fraction(4)):
        self.path = str(path)
        self.frame_rate = frame_rate

    def get_stream_source(self):
        return self.path

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_frame_rate(self):
        return self.frame_rate

    def get_duration(self):
        with av.open(self.path) as container:
            return float(container.duration / av.time_base)


def _write_test_video(path, frame_count=4, width=16, height=12):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=4)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        for index in range(frame_count):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[..., index % 3] = 255
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def _write_test_video_with_audio(path):
    sample_rate = 48_000
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("h264", rate=4)
        video_stream.width = 16
        video_stream.height = 12
        video_stream.pix_fmt = "yuv420p"
        video_stream.codec_context.max_b_frames = 0
        audio_stream = container.add_stream("aac", rate=sample_rate, layout="mono")
        for index in range(4):
            rgb = np.full((12, 16, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode(None))
        written = 0
        while written < sample_rate:
            samples = min(1024, sample_rate - written)
            waveform = np.zeros((1, samples), dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(
                waveform,
                format="fltp",
                layout="mono",
            )
            frame.sample_rate = sample_rate
            frame.pts = written
            frame.time_base = Fraction(1, sample_rate)
            written += samples
            container.mux(audio_stream.encode(frame))
        container.mux(audio_stream.encode(None))


def test_parse_hex_color_accepts_hash_and_rejects_invalid_values():
    assert parse_hex_color("#808080") == (128, 128, 128)
    assert parse_hex_color("12aBcF") == (0x12, 0xAB, 0xCF)
    with pytest.raises(ValueError, match="six-digit"):
        parse_hex_color("grey")


def test_native_video_pipeline_is_file_backed_and_encodes_solid_background(tmp_path):
    source_path = tmp_path / "source.mp4"
    tracking_path = tmp_path / "tracking.frames"
    alpha_path = tmp_path / "alpha.frames"
    video_only_path = tmp_path / "video-only.mp4"
    output_path = tmp_path / "output.mp4"
    matte_output_path = tmp_path / "matte.mkv"
    _write_test_video(source_path)
    video = FakeNativeVideo(source_path)

    frames, info = prepare_tracking_frames(
        video,
        tracking_path,
        image_size=8,
        kind="sam2",
    )
    assert info.frame_count == 4
    assert (info.width, info.height) == (16, 12)
    assert tracking_path.stat().st_size > 0
    assert tuple(frames[0].shape) == (3, 8, 8)

    alphas = DiskAlphaStore(alpha_path, 4, 12, 16)
    for index in range(4):
        alphas.write(index, torch.zeros(12, 16))
    alphas.flush()
    encoder = encode_background_video(
        video,
        alphas,
        info,
        background=(128, 128, 128),
        output_path=output_path,
        video_only_path=video_only_path,
        matte_output_path=matte_output_path,
        preserve_audio=False,
        encoder="libx264",
        worker_threads=2,
        pipeline_depth=2,
    )
    assert encoder == "libx264"

    with av.open(str(output_path)) as container:
        decoded = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    assert len(decoded) == 4
    assert np.asarray(decoded).mean() == pytest.approx(128, abs=5)

    with av.open(str(matte_output_path)) as container:
        matte_frames = [
            frame.to_ndarray(format="gray") for frame in container.decode(video=0)
        ]
    assert len(matte_frames) == 4
    assert np.asarray(matte_frames).mean() == 0

    alphas.close()
    frames.close()


def test_tracking_cache_modes_include_lossless_default_and_jpeg_opt_in(tmp_path):
    source_path = tmp_path / "source.mp4"
    _write_test_video(source_path)
    video = FakeNativeVideo(source_path)
    sequences = {}

    for cache_mode in TRACKING_CACHE_MODES:
        sequence, info = prepare_tracking_frames(
            video,
            tmp_path / f"tracking-{cache_mode}.frames",
            image_size=8,
            kind="sam2",
            cache_mode=cache_mode,
            worker_threads=2,
            pipeline_depth=2,
        )
        assert info.frame_count == 4
        sequences[cache_mode] = sequence

    assert torch.equal(sequences["lossless_zstd"][0], sequences["raw_fast"][0])
    assert tuple(sequences["jpeg_low_disk"][0].shape) == (3, 8, 8)
    for sequence in sequences.values():
        sequence.close()


def test_alpha_store_async_writes_are_bounded_and_random_access(tmp_path):
    store = DiskAlphaStore(
        tmp_path / "alpha-async.frames",
        frame_count=6,
        height=12,
        width=16,
        worker_threads=3,
        queue_depth=2,
    )
    for index in reversed(range(6)):
        store.write(index, torch.full((12, 16), index / 5))
    store.flush()

    for index in range(6):
        assert store.as_float(index).mean() == pytest.approx(index / 5, abs=1 / 255)
    store.close()


def test_temporal_alpha_stabilization_suppresses_one_frame_excursions(tmp_path):
    store = DiskAlphaStore(
        tmp_path / "alpha-stabilize.frames",
        frame_count=5,
        height=4,
        width=4,
    )
    for index, opacity in enumerate((1.0, 1.0, 0.0, 1.0, 1.0)):
        store.write(index, torch.full((4, 4), opacity))
    store.flush()

    raw = _temporally_stabilized_alpha(store, 2, 5, strength=0.0)
    gentle = _temporally_stabilized_alpha(store, 2, 5, strength=0.35)
    full = _temporally_stabilized_alpha(store, 2, 5, strength=1.0)

    assert raw.mean() == pytest.approx(0.0)
    assert gentle.mean() == pytest.approx(0.35)
    assert full.mean() == pytest.approx(1.0)
    assert _temporally_stabilized_alpha(store, 0, 5, strength=1.0).mean() == 1.0
    store.close()


def test_native_video_pipeline_preserves_audio_without_buffering_a_tensor(tmp_path):
    source_path = tmp_path / "source-audio.mp4"
    _write_test_video_with_audio(source_path)
    video = FakeNativeVideo(source_path)
    frames, info = prepare_tracking_frames(
        video,
        tmp_path / "tracking.frames",
        image_size=8,
        kind="sam2",
    )
    alphas = DiskAlphaStore(tmp_path / "alpha.frames", 4, 12, 16)
    for index in range(4):
        alphas.write(index, torch.ones(12, 16))
    alphas.flush()
    output_path = tmp_path / "output-audio.mp4"

    encode_background_video(
        video,
        alphas,
        info,
        background=(128, 128, 128),
        output_path=output_path,
        video_only_path=tmp_path / "video-only.mp4",
        preserve_audio=True,
    )

    with av.open(str(output_path)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert sum(frame.samples for frame in container.decode(audio=0)) > 0

    alphas.close()
    frames.close()


def test_streaming_matte_video_preserves_soft_alpha_losslessly(tmp_path):
    source_path = tmp_path / "source.mp4"
    _write_test_video(source_path)
    video = FakeNativeVideo(source_path)
    frames, info = prepare_tracking_frames(
        video,
        tmp_path / "tracking.frames",
        image_size=8,
        kind="sam2",
    )
    alphas = DiskAlphaStore(tmp_path / "alpha.frames", 4, 12, 16)
    for index, value in enumerate((0, 64, 128, 255)):
        alphas.write(index, torch.full((12, 16), value / 255.0))
    alphas.flush()
    matte_output_path = tmp_path / "matte.mkv"

    encode_background_video(
        video,
        alphas,
        info,
        background=(128, 128, 128),
        output_path=tmp_path / "output.mp4",
        video_only_path=tmp_path / "video-only.mp4",
        matte_output_path=matte_output_path,
        preserve_audio=False,
        encoder="libx264",
        edge_stabilization=0.0,
    )

    with av.open(str(matte_output_path)) as container:
        decoded = [
            frame.to_ndarray(format="gray") for frame in container.decode(video=0)
        ]
    assert [int(frame.mean()) for frame in decoded] == [0, 64, 128, 255]

    alphas.close()
    frames.close()


@pytest.mark.parametrize(("output_fps", "expected_frames"), [(2.0, 2), (8.0, 8)])
def test_native_video_pipeline_converts_fps_without_changing_duration(
    tmp_path, output_fps, expected_frames
):
    source_path = tmp_path / "source.mp4"
    _write_test_video(source_path)
    video = FakeNativeVideo(source_path)
    frames, info = prepare_tracking_frames(
        video,
        tmp_path / "tracking.frames",
        image_size=8,
        kind="sam2",
    )
    alphas = DiskAlphaStore(tmp_path / "alpha.frames", 4, 12, 16)
    for index in range(4):
        alphas.write(index, torch.ones(12, 16))
    alphas.flush()
    output_path = tmp_path / f"output-{output_fps}.mp4"

    encode_background_video(
        video,
        alphas,
        info,
        background=(128, 128, 128),
        output_path=output_path,
        video_only_path=tmp_path / f"video-only-{output_fps}.mp4",
        output_fps=output_fps,
        preserve_audio=False,
    )

    with av.open(str(output_path)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
        assert Fraction(stream.average_rate) == Fraction(int(output_fps))
    assert len(decoded) == expected_frames
    assert len(decoded) / output_fps == pytest.approx(1.0)

    alphas.close()
    frames.close()
