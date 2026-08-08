import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from hydra import initialize_config_dir

from video_model import (
    SAM2_VENDOR_PACKAGE,
    SAM2MattingVideoModel,
    VENDOR_DIR,
    _activate_vendored_package,
    _extract_propagated_alpha,
    make_checkerboard_preview,
    prepare_seed_mask,
    select_sam3_text_mask,
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


class FakeMovableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))


def test_vendored_sam2_has_private_namespace_and_coexists_with_installed_sam2(
    monkeypatch,
):
    external_sam2 = types.ModuleType("sam2")
    external_sam2.__file__ = "/external/site-packages/sam2/__init__.py"
    monkeypatch.setitem(sys.modules, "sam2", external_sam2)

    _activate_vendored_package(SAM2_VENDOR_PACKAGE)
    private_sam2 = importlib.import_module(SAM2_VENDOR_PACKAGE)
    importlib.import_module(f"{SAM2_VENDOR_PACKAGE}.sam2matting_video_predictor")

    assert sys.modules["sam2"] is external_sam2
    assert Path(private_sam2.__file__).resolve().is_relative_to(VENDOR_DIR)


def test_private_sam2_config_ignores_an_existing_hydra_search_path(tmp_path):
    _activate_vendored_package(SAM2_VENDOR_PACKAGE)
    from comfyui_sam2matting_sam2.build_sam import _load_config

    with initialize_config_dir(config_dir=str(tmp_path), version_base="1.2"):
        cfg = _load_config(
            "configs/sam2matting-sam2.1tiny.yaml",
            ["++model.fill_hole_area=0"],
        )

    assert cfg.model.fill_hole_area == 0
    assert cfg.model._target_.startswith(f"{SAM2_VENDOR_PACKAGE}.")


def test_sam2_matting_head_uses_cuda_ready_cached_frame_not_offloaded_source():
    _activate_vendored_package(SAM2_VENDOR_PACKAGE)
    from comfyui_sam2matting_sam2.sam2matting_video_predictor import (
        SAM2VideoPredictor,
    )

    class FakePredictor:
        fill_hole_area = 0

        def __init__(self):
            self.active_image = torch.ones(1, 3, 4, 6)
            self.alpha_image = None

        def _get_image_feature(self, state, frame_idx, batch_size):
            return (
                self.active_image,
                {"backbone_fpn": [torch.ones(1, 1, 1, 1)]},
                [],
                [],
                [],
            )

        def track_step(self, **kwargs):
            return {
                "maskmem_features": None,
                "maskmem_pos_enc": [torch.zeros(1, 1, 1, 1)],
                "pred_masks": torch.ones(1, 1, 2, 2),
                "obj_ptr": torch.zeros(1, 1),
                "object_score_logits": torch.zeros(1, 1),
            }

        def _forward_alpha_heads(self, *, image, **kwargs):
            self.alpha_image = image
            alpha = torch.ones(1, 1, 2, 2)
            return alpha, alpha, None

        def _get_maskmem_pos_enc(self, state, current_out):
            return current_out["maskmem_pos_enc"]

    predictor = FakePredictor()
    state = {
        "images": [torch.zeros(3, 4, 6)],
        "storage_device": torch.device("cpu"),
        "num_frames": 1,
        "video_height": 4,
        "video_width": 6,
    }

    SAM2VideoPredictor._run_single_frame_inference(
        predictor,
        inference_state=state,
        output_dict={},
        frame_idx=0,
        batch_size=1,
        is_init_cond_frame=True,
        point_inputs=None,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
    )

    assert predictor.alpha_image is predictor.active_image


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


def test_sam3_text_mask_selects_highest_scoring_detection():
    masks = torch.zeros(2, 1, 3, 4, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 1:, 2:] = True

    selected, score = select_sam3_text_mask(
        masks,
        torch.tensor([0.61, 0.92]),
        "highest_score",
    )

    assert selected.shape == (3, 4)
    assert selected.sum().item() == 4
    assert score == pytest.approx(0.92)


def test_sam3_text_mask_can_combine_all_detections():
    masks = torch.zeros(2, 1, 2, 3, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 1, 2] = True

    selected, score = select_sam3_text_mask(
        masks,
        torch.tensor([0.8, 0.7]),
        "combine_all",
    )

    assert selected.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert score == pytest.approx(0.8)


def test_sam3_text_mask_reports_when_prompt_finds_nothing():
    with pytest.raises(RuntimeError, match="did not find an object"):
        select_sam3_text_mask(
            torch.zeros(0, 1, 2, 2),
            torch.zeros(0),
            "highest_score",
        )


def test_sam3_text_seed_uses_selected_frame_and_returns_mask(monkeypatch):
    processor_module = types.ModuleType("sam3.model.sam3_image_processor")

    class FakeProcessor:
        def __init__(self, model, device, confidence_threshold):
            assert isinstance(model, FakeMovableModel)
            assert device == "cpu"
            assert confidence_threshold == pytest.approx(0.6)

        def set_image(self, image):
            assert image.shape == (3, 2, 4)
            assert image[0, 0, 0].item() == pytest.approx(0.75)
            return {"image": image}

        def set_text_prompt(self, prompt, state):
            assert prompt == "red car"
            state["masks"] = torch.tensor(
                [[[[False, True, True, False]] * 2]], dtype=torch.bool
            )
            state["scores"] = torch.tensor([0.88])
            return state

    processor_module.Sam3Processor = FakeProcessor
    sam3_package = types.ModuleType("sam3")
    sam3_package.__path__ = []
    sam3_model_package = types.ModuleType("sam3.model")
    sam3_model_package.__path__ = []
    monkeypatch.setitem(sys.modules, "sam3", sam3_package)
    monkeypatch.setitem(sys.modules, "sam3.model", sam3_model_package)
    monkeypatch.setitem(
        sys.modules,
        "sam3.model.sam3_image_processor",
        processor_module,
    )

    model = SAM2MattingVideoModel.from_predictor(
        "sam3", FakeMovableModel(), device="cpu"
    )
    model.checkpoint_path = "unused-in-test"
    model._sam3_text_model = FakeMovableModel()
    images = torch.zeros(2, 2, 4, 3)
    images[1, ..., 0] = 0.75

    mask, score = model.text_seed_mask(
        images,
        " red car ",
        frame_index=1,
        confidence_threshold=0.6,
    )

    assert mask.shape == (1, 2, 4)
    assert mask[0, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert score == pytest.approx(0.88)


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
