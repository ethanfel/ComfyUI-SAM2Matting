# ComfyUI-SAM2Matting

Video-first ComfyUI nodes for
[FudanCVL SAM2Matting](https://github.com/FudanCVL/SAM2Matting). One ComfyUI
`IMAGE` batch is treated as one ordered clip: the node creates one tracker
state, applies a foreground mask on a selected frame, and propagates that state
through the complete sequence. It does **not** run the image predictor on each
frame independently.

## Nodes

### Load SAM2Matting Video Model

- `variant`
  - `sam2.1_base_plus` — default quality/speed balance
  - `sam2.1_tiny` — fastest and lowest VRAM
  - `sam3` — largest model and highest dependency/compute cost
- `compile_model` — opt-in image-backbone compilation; the first run can take a
  long time

Checkpoints download on first use to `ComfyUI/models/sam2matting/`:

- `SAM2Matting-SAM2.1Base+.pt`
- `SAM2Matting-SAM2.1Tiny.pt`
- `SAM2Matting-SAM3.pt`

### SAM2Matting Video

Inputs:

- `model` — loaded video model
- `images` — one ordered `IMAGE` batch representing one clip
- `initial_mask` — one foreground `MASK`, or a mask batch matching the frame
  count
- `mask_frame` — frame receiving the seed mask
- `mask_threshold` — threshold applied after resizing the mask; white is
  foreground
- `memory_mode`
  - `balanced` — source frames stay on CPU; temporal state stays on the model
    device
  - `low_vram` — source frames and temporal state are offloaded to CPU
  - `maximum_speed` — preprocessed source frames and temporal state stay on the
    model device

Outputs:

- `alpha` — ordered `[frames, height, width]` `MASK` batch in `[0, 1]`
- `foreground_rgb` — the original, unpremultiplied RGB frames
- `preview` — RGB composited over a checkerboard

The foreground output intentionally remains unpremultiplied. Connect
`foreground_rgb` and `alpha` to ComfyUI's alpha-join/compositing nodes. Encoding,
frame rate, and audio remain the responsibility of Video Helper Suite or another
video workflow.

## Installation

Place this repository under `ComfyUI/custom_nodes/`, then install its Python
dependencies with the same Python environment that runs ComfyUI:

```bash
python -m pip install -r requirements.txt
```

Restart ComfyUI. The nodes are in `SAM2Matting/video`.

Do not install the official project's pinned `torch`, `torchvision`, or
`torch-tensorrt` versions over ComfyUI's environment. This package deliberately
leaves torch version management to ComfyUI.

## Typical workflow

1. Load or decode a video as an ordered `IMAGE` batch.
2. Produce or paint a rough white-foreground mask for one frame.
3. Set `mask_frame` to that frame's zero-based index.
4. Run `SAM2Matting Video`.
5. Inspect `preview`, then combine `foreground_rgb` with `alpha` and encode the
   result using your existing video nodes.

If `initial_mask` contains one mask, it seeds `mask_frame`. If its batch length
matches the video, the mask at `mask_frame` is selected. Other mask batch sizes
are rejected rather than silently repeated.

## Temporal and memory behavior

- A nonzero `mask_frame` is propagated forward and backward so every source
  frame receives an alpha matte.
- Frames are adapted directly from ComfyUI tensors. `balanced` and `low_vram`
  resize and normalize frames lazily, avoiding a second full preprocessed copy
  of a long clip.
- Each execution owns a fresh tracker state, and the state is reset and cleared
  on completion, interruption, or failure.
- A loaded predictor is locked during propagation so two workflows cannot
  accidentally share mutable temporal state.
- Chunking is intentionally absent: independent chunks would create temporal
  discontinuities. Split a clip only when you are prepared to reseed it.

## Current scope and limitations

- Single object, seeded by a mask. Point, box, multiple-object, and SAM3 text
  prompts are not exposed yet.
- All frames in one `IMAGE` tensor necessarily have the same resolution.
- SAM3 follows the current upstream implementation and requires CUDA.
- SAM2/SAM3 use top-level Python package names upstream. If another custom node
  has already loaded a different incompatible `sam2` or `sam3` implementation,
  this node stops with an explicit collision error instead of mixing code.
- Checkpoint quality and performance claims are upstream claims until reproduced
  in your own ComfyUI environment.

## Tests

The focused tests use a fake temporal predictor, so they do not download a
checkpoint or require a GPU:

```bash
pytest -q
```

They cover mask polarity and frame selection, forward/reverse propagation using
one state, ordered alpha assembly, SAM2/SAM3 return-shape normalization,
cleanup, and preview tensor contracts.

## License

The official SAM2Matting repository states **CC BY-NC-SA 4.0** and limits use to
non-commercial research. Its license is included at
`vendor/SAM2MATTING_LICENSE`; see `THIRD_PARTY_NOTICES.md`. Review the upstream
SAM2 and SAM3 terms as well before redistributing or deploying this package.
