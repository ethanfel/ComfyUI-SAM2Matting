# ComfyUI-SAM2Matting

Video matting for ComfyUI using
[FudanCVL SAM2Matting](https://github.com/FudanCVL/SAM2Matting).

Give the node a video and one rough foreground mask. It tracks the selected
subject through the clip and returns a soft alpha matte.

## Which model should I use?

| Model | Best for | System RAM | Peak VRAM, 720p / 1080p | Speed, 720p / 1080p | Included here |
| --- | --- | ---: | ---: | ---: | --- |
| `sam2.1_tiny` | Fast previews, any subject | 16 GB+ | 3.08 / 3.61 GB | 40.46 / 40.31 FPS | Yes |
| `sam2.1_base_plus` | Best default, any subject | 16 GB+ | 3.42 / 3.82 GB | 30.40 / 30.36 FPS | Yes, default |
| `sam3` | Text-guided masks, any subject | 24 GB+ | 4.80 / 4.91 GB | 9.09 / 9.07 FPS | Yes |
| [MatAnyone2](https://github.com/pq-yang/MatAnyone2) | Dedicated human matting | 16 GB+ | 3.10 / 13.67 GB | 21.94 / 9.93 FPS | No |

VRAM and speed are the upstream
[SAM2Matting paper's](https://arxiv.org/abs/2606.27339) VideoMatte results on
one NVIDIA A6000, not measurements from this ComfyUI node. Real performance
depends on the GPU, clip, PyTorch build, and memory mode.

System RAM values are conservative starting points for the tensor workflow.
Clip length is the main factor there: one decoded float32 RGB frame uses about
10.5 MiB at 720p or 23.7 MiB at 1080p, before mattes and other ComfyUI nodes.
The streaming background node described below avoids that full-frame batch.

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

For removing a background and replacing it with a solid grey:

1. Load the clip with ComfyUI's native **Load Video** node.
2. Paint or generate one black-and-white seed mask. White is foreground.
3. Load `sam2.1_base_plus` with **Load SAM2Matting Video Model**.
4. Connect everything to **SAM2Matting Video Background (Streaming)**.
5. Leave `background_color` at `#808080`, set the matching zero-based
   `mask_frame`, and connect its native `VIDEO` output to **Save Video**.

Drag
[`example_workflows/sam2matting_video_background_streaming.json`](example_workflows/sam2matting_video_background_streaming.json)
onto ComfyUI for this setup. It never converts the complete clip to an `IMAGE`
or `MASK` batch, and it preserves source audio by default.

The original tensor node remains useful when the alpha matte must feed other
ComfyUI image nodes. Load the video as one ordered `IMAGE` batch, then connect
it and the seed mask to **SAM2Matting Video**.

With SAM3, **SAM3 Text Prompt to Seed Mask** can replace the painted mask:

1. Connect the loaded SAM3 model and video batch to the prompt node.
2. Describe the subject and choose the matching `frame_index`.
3. Connect `seed_mask` to `initial_mask` and `mask_frame` to `mask_frame` on
   **SAM2Matting Video**.
4. Inspect the prompt node's preview before running the complete clip.

For the tensor/transparent-output example, drag
[`example_workflows/sam2matting_video_default.json`](example_workflows/sam2matting_video_default.json)
onto ComfyUI. It uses
[Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
and produces a transparent VP9 WebM plus a black-background/white-foreground
matte video. The original video batch is reused for the RGB output, so the
matting node only adds its alpha batch to ComfyUI's cache.

For text-prompt selection, use
[`example_workflows/sam3_text_prompt_video.json`](example_workflows/sam3_text_prompt_video.json).
It loads SAM3, turns a prompt such as `person` into a seed on one selected
frame, shows that single-frame preview, and passes both the mask and matching
frame index into temporal matting. The temporal node returns only its alpha
batch.

Both examples are capped at 48 frames for a quick first test. Set
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

To create transparent output, connect the original video batch directly to
**Join Image with Alpha**. ComfyUI's **Join Image with Alpha** uses inverse
`MASK` semantics, so pass `alpha` through **Invert Mask** before connecting it
to the alpha input, as shown in the example workflow.

The node deliberately does not return copied RGB frames or a full checkerboard
preview. ComfyUI caches every returned tensor, including unconnected outputs;
removing those two outputs cuts this node's cached result from seven float
channels per pixel to one. The input video and downstream output nodes can
still be retained by ComfyUI's own cache. Use ComfyUI's `--cache-none` option
when no cross-run cache retention is desired.

### SAM2Matting Video Background (Streaming)

This is the recommended node when the final result is a normal video over one
solid background color.

- `video`: native ComfyUI `VIDEO`; do not place **Get Video Components** before
  it
- `initial_mask`: one white-foreground seed mask
- `mask_frame`: the source frame matching the seed
- `background_color`: six-digit RGB hex, default `#808080`
- `state_device`:
  - `gpu`: recommended; keeps the tracker's temporal state out of system RAM
  - `cpu`: lowers VRAM use but temporal state grows in system RAM
- `crf`: H.264 quality; lower is higher quality and larger
- `preserve_audio`: transcodes the active source audio to AAC

The output is a file-backed native `VIDEO`, ready for ComfyUI's **Save Video**.
During execution, model-resolution tracking frames are kept as compressed JPEG
records and soft mattes as lossless 8-bit PNG records in ComfyUI's temporary
directory. Only the current full-resolution source frame is composited in RAM.
The temporary files are removed after encoding; the file backing the returned
`VIDEO` remains until ComfyUI cleans its normal temporary directory.

This trades temporary disk I/O and two sequential video decodes for predictable
host memory. Temporary disk use still grows with clip length and image content.
The temporal tracker state also grows with clip length on the selected
`state_device`; the node does not independently batch or reset the tracker,
because doing so would break temporal continuity.

### SAM3 Text Prompt to Seed Mask

This optional node requires the loader variant `sam3`.

- `images`: the same video batch used for matting
- `text_prompt`: a short subject description, such as `woman in red jacket`
- `frame_index`: the frame on which SAM3 should find the subject
- `confidence_threshold`: removes lower-confidence detections
- `selection`:
  - `highest_score`: use the best matching detection
  - `combine_all`: merge every detection above the threshold

It outputs the seed `MASK`, a checkerboard preview, the highest detection score,
and the matching frame index. The detector is cached in system RAM and swapped
onto the GPU only while generating a seed mask, avoiding a permanent second
SAM3 vision backbone in VRAM. The first prompt is therefore slower.

## Practical limits

- One tracked object per run, seeded by a mask.
- The tensor node processes the whole frame batch as one temporal clip.
- A nonzero `mask_frame` propagates both forward and backward.
- For long solid-background renders, use the streaming node with
  `state_device=gpu`. The tensor node and `state_device=cpu` can still consume
  substantial system RAM as the clip grows.
- SAM3 requires CUDA.
- Point, box, and multi-object propagation are not exposed yet.
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
