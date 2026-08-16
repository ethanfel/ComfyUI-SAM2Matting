import ast
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATHS = (
    REPO_ROOT / "example_workflows" / "sam2matting_video_default.json",
    REPO_ROOT / "example_workflows" / "sam3_text_prompt_video.json",
    REPO_ROOT / "example_workflows" / "sam2matting_video_background_streaming.json",
)

TENSOR_WORKFLOW_PATHS = WORKFLOW_PATHS[:2]


def _load_workflow(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=lambda path: path.stem)
def test_example_workflow_links_are_consistent(path):
    workflow = _load_workflow(path)
    nodes = {node["id"]: node for node in workflow["nodes"]}

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow[
        "links"
    ]:
        source_output = nodes[source_id]["outputs"][source_slot]
        target_input = nodes[target_id]["inputs"][target_slot]

        assert link_id in (source_output["links"] or [])
        assert target_input["link"] == link_id
        assert source_output["type"] == link_type
        assert target_input["type"] == link_type


@pytest.mark.parametrize("path", TENSOR_WORKFLOW_PATHS, ids=lambda path: path.stem)
def test_example_workflow_reuses_video_and_matting_returns_only_alpha(path):
    workflow = _load_workflow(path)
    matting = _node(workflow, 4)
    join_alpha = _node(workflow, 6)

    assert matting["type"] == "SAM2MattingVideo"
    assert [output["name"] for output in matting["outputs"]] == ["alpha"]

    image_link_id = join_alpha["inputs"][0]["link"]
    image_link = next(link for link in workflow["links"] if link[0] == image_link_id)
    assert image_link[1:3] == [2, 0]


def test_streaming_workflow_never_converts_native_video_to_an_image_batch():
    workflow = _load_workflow(WORKFLOW_PATHS[2])
    node_types = {node["type"] for node in workflow["nodes"]}
    streaming = _node(workflow, 4)

    assert {"LoadVideo", "SAM2MattingVideoBackground", "SaveVideo"} <= node_types
    assert "GetVideoComponents" not in node_types
    assert [output["type"] for output in streaming["outputs"]] == ["VIDEO"]


def test_matting_node_api_exposes_only_alpha():
    module = ast.parse((REPO_ROOT / "nodes.py").read_text(encoding="utf-8"))
    matting_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SAM2MattingVideo"
    )
    assignments = {
        target.id: ast.literal_eval(statement.value)
        for statement in matting_class.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id in {"RETURN_TYPES", "RETURN_NAMES"}
    }

    assert assignments == {
        "RETURN_TYPES": ("MASK",),
        "RETURN_NAMES": ("alpha",),
    }
