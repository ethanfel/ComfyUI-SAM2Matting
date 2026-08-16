import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from hydra import initialize_config_dir

from video_model import (
    SAM2_VENDOR_PACKAGE,
    SAM3_VENDOR_PACKAGE,
    SAM2MattingVideoModel,
    VENDOR_DIR,
    _activate_vendored_package,
    _extract_propagated_alpha,
    _prune_temporal_state,
    _temporal_state_horizon,
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


def test_vendored_sam3_has_private_namespace_and_wins_path_precedence(
    monkeypatch, tmp_path
):
    external_sam3 = types.ModuleType("sam3")
    external_sam3.__file__ = "/external/site-packages/sam3/__init__.py"
    monkeypatch.setitem(sys.modules, "sam3", external_sam3)

    decoy_root = tmp_path / "decoy"
    decoy_package = decoy_root / SAM3_VENDOR_PACKAGE
    decoy_package.mkdir(parents=True)
    (decoy_package / "__init__.py").write_text("SOURCE = 'decoy'\n")
    vendor_path = str(VENDOR_DIR)
    monkeypatch.setattr(
        sys,
        "path",
        [str(decoy_root), vendor_path, *sys.path],
    )

    _activate_vendored_package(SAM3_VENDOR_PACKAGE)
    spec = importlib.machinery.PathFinder.find_spec(SAM3_VENDOR_PACKAGE, sys.path)

    assert sys.modules["sam3"] is external_sam3
    assert sys.path[0] == vendor_path
    assert sys.path.count(vendor_path) == 1
    assert spec is not None
    assert Path(spec.origin).resolve().is_relative_to(VENDOR_DIR)
    assert (
        VENDOR_DIR
        / SAM3_VENDOR_PACKAGE
        / "model"
        / "sam3matting_video_predictor.py"
    ).is_file()


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
    processor_module = types.ModuleType(
        f"{SAM3_VENDOR_PACKAGE}.model.sam3_image_processor"
    )

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
    sam3_package = types.ModuleType(SAM3_VENDOR_PACKAGE)
    sam3_package.__path__ = []
    sam3_model_package = types.ModuleType(f"{SAM3_VENDOR_PACKAGE}.model")
    sam3_model_package.__path__ = []
    monkeypatch.setitem(sys.modules, SAM3_VENDOR_PACKAGE, sam3_package)
    monkeypatch.setitem(
        sys.modules,
        f"{SAM3_VENDOR_PACKAGE}.model",
        sam3_model_package,
    )
    monkeypatch.setitem(
        sys.modules,
        f"{SAM3_VENDOR_PACKAGE}.model.sam3_image_processor",
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


def test_file_backed_api_emits_each_alpha_without_building_a_batch():
    predictor = FakeSAM2Predictor()
    model = SAM2MattingVideoModel.from_predictor(
        "sam2.1_tiny", predictor, device="cpu"
    )
    frames = [torch.zeros(3, 8, 8) for _ in range(3)]
    emitted = {}

    model.matte_frame_sequence(
        frames,
        height=4,
        width=6,
        initial_mask=torch.ones(4, 6),
        mask_frame=1,
        alpha_callback=lambda index, alpha: emitted.__setitem__(index, alpha),
    )

    assert sorted(emitted) == [0, 1, 2]
    assert all(tuple(alpha.shape) == (4, 6) for alpha in emitted.values())
    assert predictor.reset_called


def test_temporal_state_horizon_covers_mask_memory_and_object_pointers():
    predictor = types.SimpleNamespace(
        num_maskmem=7,
        memory_temporal_stride_for_eval=1,
        max_obj_ptrs_in_encoder=16,
    )
    assert _temporal_state_horizon(predictor) == 15

    predictor.memory_temporal_stride_for_eval = 5
    assert _temporal_state_horizon(predictor) == 26


def test_bounded_sam2_state_keeps_seed_context_and_recent_forward_history():
    predictor = types.SimpleNamespace(
        num_maskmem=7,
        memory_temporal_stride_for_eval=1,
        max_obj_ptrs_in_encoder=16,
    )
    outputs = {index: object() for index in range(10, 101)}
    tracked = {index: {"reverse": False} for index in outputs}
    state = {
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": outputs,
            }
        },
        "frames_tracked_per_obj": {0: tracked},
    }

    _prune_temporal_state(
        predictor,
        state,
        current_frame=100,
        mask_frame=10,
        reverse=False,
    )

    expected = set(range(10, 26)) | set(range(85, 101))
    assert set(outputs) == expected
    assert set(tracked) == expected


def test_bounded_sam3_state_drops_shared_maps_behind_reverse_window():
    predictor = types.SimpleNamespace(
        num_maskmem=7,
        memory_temporal_stride_for_eval=1,
        max_obj_ptrs_in_encoder=16,
    )
    global_outputs = {index: object() for index in range(101)}
    per_object_outputs = {index: object() for index in range(101)}
    tracked = {index: {"reverse": index <= 50} for index in range(101)}
    state = {
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": global_outputs,
        },
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": per_object_outputs,
            }
        },
        "frames_already_tracked": tracked,
    }

    _prune_temporal_state(
        predictor,
        state,
        current_frame=20,
        mask_frame=50,
        reverse=True,
    )

    expected = set(range(36))
    assert set(global_outputs) == expected
    assert set(per_object_outputs) == expected
    assert set(tracked) == expected


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
