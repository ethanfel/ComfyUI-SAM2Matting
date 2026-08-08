# ComfyUI-SAM2Matting

Video matting for ComfyUI using
[FudanCVL SAM2Matting](https://github.com/FudanCVL/SAM2Matting).

Give the node a video and one rough foreground mask. It tracks the selected
subject through the clip and returns a soft alpha matte, the original RGB
frames, and a checkerboard preview.

## Which model should I use?

| Model | Best for | Target | Trade-off | Included here |
| --- | --- | --- | --- | --- |
| `sam2.1_tiny` | Fast previews and lower VRAM use | Open-world subjects | Smallest SAM2Matting option | Yes |
| `sam2.1_base_plus` | Most workflows | Open-world subjects | Recommended quality/speed balance | Yes, default |
| `sam3` | Trying the largest upstream tracker | Open-world subjects | Highest compute cost; CUDA required | Yes |
| [MatAnyone2](https://github.com/pq-yang/MatAnyone2) | Dedicated human video matting | People | Separate model and ComfyUI implementation | No |

SAM2Matting is designed for varied subjects such as people, animals, anime,
and translucent objects. MatAnyone2 is specifically presented as a human video
matting model. They have not published a direct, like-for-like comparison, so
choose by subject and workflow rather than treating this table as a quality
ranking.

## Install

Search for **SAM2Matting Video** in ComfyUI-Manager, or run:

```bash
comfy node install sam2matting-video
```

For a manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-SAM2Matting.git
cd ComfyUI-SAM2Matting
python -m pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear under `SAM2Matting/video`.

The required checkpoint downloads automatically on first use to
`ComfyUI/models/sam2matting/`. Do not replace ComfyUI's existing PyTorch
installation with the versions pinned by the upstream research repository.

## Quick start

1. Load a video as one ordered ComfyUI `IMAGE` batch.
2. Paint or generate a black-and-white mask for one frame. White is foreground.
3. Load `sam2.1_base_plus` with **Load SAM2Matting Video Model**.
4. Connect the video and mask to **SAM2Matting Video**.
5. Set `mask_frame` to the zero-based frame matching the mask, then run.

For a ready-made example, drag
[`example_workflows/sam2matting_video_default.json`](example_workflows/sam2matting_video_default.json)
onto ComfyUI. It uses
[Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
and produces matte previews, a checkerboard preview video, and a transparent
VP9 WebM.

The example is capped at 48 frames for a quick first test. Set
`frame_load_cap` to `0` to process the complete video.

## Nodes

### Load SAM2Matting Video Model

- `variant`: `sam2.1_tiny`, `sam2.1_base_plus`, or `sam3`
- `compile_model`: optionally compiles the image backbone; the first run can be
  slow

### SAM2Matting Video

Inputs:

- `images`: the video frames as one ordered `IMAGE` batch
- `initial_mask`: a foreground `MASK`
- `mask_frame`: the frame that matches the seed mask
- `mask_threshold`: converts the seed into a binary tracking mask
- `memory_mode`:
  - `balanced`: recommended default
  - `low_vram`: offloads frames and temporal state to CPU
  - `maximum_speed`: keeps frames and state on the model device

Outputs:

- `alpha`: soft foreground opacity for every frame
- `foreground_rgb`: original, unpremultiplied RGB frames
- `preview`: checkerboard composite for inspection

ComfyUI's **Join Image with Alpha** uses inverse `MASK` semantics. Pass `alpha`
through **Invert Mask** before connecting it to that node, as shown in the
example workflow.

## Practical limits

- One tracked object per run, seeded by a mask.
- The whole frame batch is processed as one temporal clip.
- A nonzero `mask_frame` propagates both forward and backward.
- Long clips can use substantial RAM or VRAM; start with `balanced` or
  `low_vram`.
- SAM3 requires CUDA.
- Point, box, text, and multi-object prompts are not exposed yet.
- Independent chunking is not provided because it would break temporal
  continuity; split and reseed clips manually when needed.

## Development

Run the checkpoint-free test suite with:

```bash
pytest -q
```

## License

The upstream SAM2Matting project uses **CC BY-NC-SA 4.0** for non-commercial
research. See `vendor/SAM2MATTING_LICENSE` and `THIRD_PARTY_NOTICES.md`, and
review the upstream SAM2 and SAM3 terms before redistribution or deployment.
