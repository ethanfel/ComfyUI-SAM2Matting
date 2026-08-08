import pytest
import torch

from video_model import (
    SAM2MattingVideoModel,
    _extract_propagated_alpha,
    make_checkerboard_preview,
    prepare_seed_mask,
)


class FakeSAM2Predictor:
    image_size = 8
    device = torch.device("cpu")

    def __init__(self):
        self.propagation_calls = []
        self.seed = None
        self.reset_called = False

    def _get_image_feature(self, state, frame_idx, batch_size):
        # The real SAM2 init primes frame zero once.
        state["primed"] = (frame_idx, batch_size, state["images"][frame_idx].shape)

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.seed = (frame_idx, obj_id, mask.detach().cpu())

    def propagate_in_video(
        self,
        state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        tqdm_disable=False,
    ):
        self.propagation_calls.append((id(state), start_frame_idx, reverse))
        if reverse:
            order = range(start_frame_idx, -1, -1)
        else:
            order = range(start_frame_idx, state["num_frames"])
        for frame_index in order:
            value = (frame_index + 1) / state["num_frames"]
            alpha = torch.full(
                (1, state["video_height"], state["video_width"]), value
            )
            yield frame_index, [1], None, alpha, None

    def reset_state(self, state):
        self.reset_called = True


class FailingSAM2Predictor(FakeSAM2Predictor):
    def propagate_in_video(self, *args, **kwargs):
        raise RuntimeError("synthetic propagation failure")
        yield  # pragma: no cover - makes this a generator like the real API


def test_prepare_seed_mask_selects_matching_video_frame_and_white_is_foreground():
    masks = torch.zeros(3, 2, 2)
    masks[1, 0, 1] = 1.0

    result = prepare_seed_mask(
        masks,
        mask_frame=1,
        frame_count=3,
        height=2,
        width=2,
        threshold=0.5,
    )

    assert result.tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_temporal_model_uses_one_state_and_propagates_both_directions():
    predictor = FakeSAM2Predictor()
    model = SAM2MattingVideoModel.from_predictor(
        "sam2.1_tiny", predictor, device="cpu"
    )
    images = torch.rand(4, 3, 5, 3)
    mask = torch.zeros(3, 5)
    mask[1:, 2:] = 1.0
    progress = []

    alpha = model.matte_video(
        images,
        mask,
        mask_frame=2,
        memory_mode="balanced",
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert alpha.shape == (4, 3, 5)
    assert torch.allclose(alpha[:, 0, 0], torch.tensor([0.25, 0.5, 0.75, 1.0]))
    assert [call[2] for call in predictor.propagation_calls] == [False, True]
    assert len({call[0] for call in predictor.propagation_calls}) == 1
    assert predictor.seed[0:2] == (2, 1)
    assert predictor.seed[2].shape == (3, 5)
    assert predictor.reset_called
    assert progress[-1] == (4, 4)


def test_frame_zero_needs_only_forward_propagation():
    predictor = FakeSAM2Predictor()
    model = SAM2MattingVideoModel.from_predictor(
        "sam2.1_base_plus", predictor, device="cpu"
    )
    model.matte_video(
        torch.rand(2, 2, 3, 3),
        torch.ones(2, 3),
        mask_frame=0,
    )
    assert [call[2] for call in predictor.propagation_calls] == [False]


def test_temporal_state_is_reset_when_propagation_fails():
    predictor = FailingSAM2Predictor()
    model = SAM2MattingVideoModel.from_predictor(
        "sam2.1_tiny", predictor, device="cpu"
    )

    with pytest.raises(RuntimeError, match="synthetic propagation failure"):
        model.matte_video(
            torch.rand(2, 2, 3, 3),
            torch.ones(2, 3),
            mask_frame=0,
        )

    assert predictor.reset_called


def test_sam2_and_sam3_propagation_tuple_shapes_normalize_to_same_fields():
    sam2_alpha = torch.ones(1, 2, 2)
    sam3_alpha = torch.zeros(1, 2, 2).numpy()

    assert _extract_propagated_alpha((3, [1], None, sam2_alpha, None)) == (
        3,
        sam2_alpha,
    )
    frame, alpha = _extract_propagated_alpha(
        (4, [1], [1], None, sam3_alpha, None)
    )
    assert frame == 4
    assert alpha is sam3_alpha


def test_checkerboard_preview_preserves_tensor_contract():
    images = torch.ones(2, 4, 6, 3)
    alpha = torch.zeros(2, 4, 6)
    alpha[0] = 1.0
    preview = make_checkerboard_preview(images, alpha, cell_size=2)

    assert preview.shape == images.shape
    assert torch.equal(preview[0], images[0])
    assert 0.0 <= float(preview.min()) <= float(preview.max()) <= 1.0


def test_empty_foreground_mask_has_actionable_error():
    try:
        prepare_seed_mask(
            torch.zeros(2, 2),
            mask_frame=0,
            frame_count=1,
            height=2,
            width=2,
            threshold=0.5,
        )
    except ValueError as error:
        assert "White must mean foreground" in str(error)
    else:
        raise AssertionError("Expected an empty foreground mask to be rejected")
