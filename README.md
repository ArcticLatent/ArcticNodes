# ArcticNodes

Custom Flux-friendly nodes for ComfyUI, kept lean for tutorial use. Drop the folder into `ComfyUI/custom_nodes` and restart ComfyUI to load.

## Nodes included

- `FluxSmartResize`: Resize an `IMAGE` tensor to a Flux-safe resolution set (1M, 1.5M, 2M) with auto orientation selection and optional upscale control.
- `FluxLatentImage`: Create an empty `LATENT` at a Flux-safe resolution, choosing set, orientation, and variant index; handy to feed into ksampler inputs for text-to-image.
- `FluxPromptWithGuidance`: One-stop prompt + guidance node; encodes text with CLIP and applies `FluxGuidance` so you can plug straight into the ksampler positive input.

## Usage

1. Copy this repo into your `ComfyUI/custom_nodes` directory (or symlink it).
2. Restart ComfyUI. The nodes appear under `Arctic/Flux`.
3. Use `FluxSmartResize` to pick and resize to a target-safe resolution before Flux conditioning; `FluxLatentImage` to seed your ksampler latent; `FluxPromptWithGuidance` to combine prompt + guidance for ksampler inputs.

## Development

- Add new node classes to `arctic_nodes/nodes.py`, register them in `arctic_nodes/__init__.py`.
- Keep dependencies minimal; current nodes use only PyTorch (already bundled with ComfyUI) and the Python standard library.
- No external install step is required beyond placing the folder in `custom_nodes`.
