# ComfyUI-SAM2Matting

Video matting for ComfyUI using
[FudanCVL SAM2Matting](https://github.com/FudanCVL/SAM2Matting).

Give the node a video and one rough foreground mask. It tracks the selected
subject through the clip and returns a soft alpha matte, the original RGB
frames, and a checkerboard preview.

## Which model should I use?

| Model | Best for | System RAM | Peak VRAM, 720p / 1080p | Speed, 720p / 1080p | Included here |
| --- | --- | ---: | ---: | ---: | --- |
| `sam2.1_tiny` | Fast previews, any subject | 16 GB+ | 3.08 / 3.61 GB | 40.46 / 40.31 FPS | Yes |
| `sam2.1_base_plus` | Best default, any subject | 16 GB+ | 3.42 / 3.82 GB | 30.40 / 30.36 FPS | Yes, default |
| `sam3` | Largest tracker, any subject | 24 GB+ | 4.80 / 4.91 GB | 9.09 / 9.07 FPS | Yes |
| [MatAnyone2](https://github.com/pq-yang/MatAnyone2) | Dedicated human matting | 16 GB+ | 3.10 / 13.67 GB | 21.94 / 9.93 FPS | No |

VRAM and speed are the upstream
[SAM2Matting paper's](https://arxiv.org/abs/2606.27339) VideoMatte results on
one NVIDIA A6000, not measurements from this ComfyUI node. Real performance
depends on the GPU, clip, PyTorch build, and memory mode.

System RAM values are conservative starting points rather than benchmark
results. Clip length is the main factor: one decoded float32 RGB frame uses
about 10.5 MiB at 720p or 23.7 MiB at 1080p, before mattes, previews, and other
ComfyUI nodes. Use 32 GB or more for longer clips.

SAM2Matting is designed for varied subjects such as people, animals, anime,
and translucent objects. MatAnyone2 is specifically presented as a human video
matting model. The hardware figures compare efficiency, not matte quality, so
choose by subject and test representative footage before committing to a long
render.

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
