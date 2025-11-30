from .nodes import FluxLatentImage, FluxPromptWithGuidance, FluxSmartResize

# Maps class name to class for ComfyUI discovery.
NODE_CLASS_MAPPINGS = {
    "FluxLatentImage": FluxLatentImage,
    "FluxPromptWithGuidance": FluxPromptWithGuidance,
    "FluxSmartResize": FluxSmartResize,
}

# Human-friendly names shown in the ComfyUI node menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLatentImage": "Flux Latent Image (Arctic Latent)",
    "FluxPromptWithGuidance": "Flux Prompt with Guidance (Arctic Latent)",
    "FluxSmartResize": "Flux Smart Resize (Arctic Latent)",
}
