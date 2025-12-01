import importlib
import math

import torch
import comfy.utils as comfy_utils

try:
    import nodes as comfy_nodes  # ComfyUI core nodes (CLIPTextEncode, FluxGuidance)
except Exception:  # pragma: no cover - handled at runtime inside ComfyUI
    comfy_nodes = None

# ~1MP flux-safe-ish resolutions (multiples of 64)
RESOLUTIONS_1M = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
]

# ~1.5MP range
RESOLUTIONS_1_5M = [
    (1280, 1152),
    (1152, 1280),
    (1472, 960),
    (960, 1472),
    (1344, 1024),
    (1024, 1344),
    (1600, 896),
    (896, 1600),
]

# ~2MP range (still reasonable for Flux on most GPUs)
RESOLUTIONS_2M = [
    (1536, 1152),
    (1152, 1536),
    (1728, 1152),
    (1152, 1728),
    (1920, 1088),
    (1088, 1920),
]

RESOLUTION_SETS = {
    "1M_flux": RESOLUTIONS_1M,
    "1_5M_flux": RESOLUTIONS_1_5M,
    "2M_flux": RESOLUTIONS_2M,
}


def _resolve_flux_guidance_cls():
    """
    Locate FluxGuidance from core nodes or flux extras.
    Returns the class or None if not found.
    """
    search_modules = []
    if comfy_nodes is not None:
        search_modules.append(comfy_nodes)

    for mod_name in ("comfy_extras.nodes_flux", "nodes_flux"):
        try:
            mod = importlib.import_module(mod_name)
            search_modules.append(mod)
        except Exception:
            continue

    for mod in search_modules:
        cls = getattr(mod, "FluxGuidance", None)
        if cls is not None:
            return cls

    return None


def _orientation(w: int, h: int) -> str:
    """Classify orientation and treat near-square (within 5%) as square."""
    if w == 0 or h == 0:
        return "landscape"

    diff_ratio = abs(w - h) / max(w, h)
    if diff_ratio < 0.05:
        return "square"
    return "landscape" if w > h else "portrait"


class FluxSmartResize:
    upscale_methods = ["bilinear", "bicubic", "lanczos", "nearest-exact", "area"]

    # Flux-safe resolutions (width, height) – stored in one orientation only
    FLUX_SAFE_RESOLUTIONS = [
        (1408, 1408),
        (1728, 1152),
        (1664, 1216),
        (1920, 1088),
        (2176, 960),
        (1024, 1024),
        (1216, 832),
        (1152, 896),
        (1344, 768),
        (1536, 640),
        (320, 320),
        (384, 256),
        (448, 320),
        (448, 256),
        (576, 256),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods, {"default": "bilinear"}),
                "total_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 128.0,
                        "step": 0.01,
                        "tooltip": "Set the total megapixels (e.g., 1.0 = 1 MP)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Arctic/Flux"

    def upscale(self, image, upscale_method, total_megapixels):
        if upscale_method in ["nearest-exact", "area"]:
            raise Exception(
                f"❌ '{upscale_method}' gives poor results.\n\n"
                f"👉 Go to the Flux Smart Resize (Arctic Latent) node and switch to another like 'lanczos'.\n\n"
                f"Node may be hidden behind KSampler."
            )

        b, h, w, c = image.shape

        # Skip scaling if image matches any Flux-safe resolution
        if (w, h) in self.FLUX_SAFE_RESOLUTIONS or (h, w) in self.FLUX_SAFE_RESOLUTIONS:
            return (image,)

        samples = image.movedim(-1, 1)
        orig_h, orig_w = samples.shape[2], samples.shape[3]

        target_pixels = int(round(total_megapixels * 1024 * 1024))
        scale_by = math.sqrt(target_pixels / (orig_w * orig_h))

        new_w = max(1, round(orig_w * scale_by))
        new_h = max(1, round(orig_h * scale_by))

        scaled = comfy_utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        scaled = scaled.movedim(1, -1)
        return (scaled,)


class FluxLatentImage:
    """
    Create an empty latent tensor sized to a Flux-safe resolution for text-to-image.
    Select from preset sets/orientations or pick an exact Flux resolution, with optional batching.
    """

    RESOLUTION_CHOICES = {
        "landscape": sorted(
            {
                (w, h)
                for res_list in RESOLUTION_SETS.values()
                for (w, h) in res_list
                if _orientation(w, h) == "landscape"
            }
        ),
        "portrait": sorted(
            {
                (w, h)
                for res_list in RESOLUTION_SETS.values()
                for (w, h) in res_list
                if _orientation(w, h) == "portrait"
            }
        ),
        "square": sorted(
            {
                (w, h)
                for res_list in RESOLUTION_SETS.values()
                for (w, h) in res_list
                if _orientation(w, h) == "square"
            }
        ),
    }

    RESOLUTION_LABELS = ["auto_from_set"] + [
        f"Landscape: {w}x{h}" for (w, h) in RESOLUTION_CHOICES["landscape"]
    ] + [
        f"Portrait: {w}x{h}" for (w, h) in RESOLUTION_CHOICES["portrait"]
    ] + [
        f"Square: {w}x{h}" for (w, h) in RESOLUTION_CHOICES["square"]
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_set": (
                    ["1M_flux", "1_5M_flux", "2M_flux"],
                    {"default": "1M_flux"},
                ),
                "orientation": (
                    ["landscape", "portrait", "square"],
                    {"default": "landscape"},
                ),
                "resolution_choice": (
                    cls.RESOLUTION_LABELS,
                    {
                        "default": "auto_from_set",
                        "label": "Resolution choice (auto or pick)",
                        "display": "dropdown",
                    },
                ),
                "variant_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 99,
                        "step": 1,
                        "label": "Variant (cycles within matches)",
                    },
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "build"
    CATEGORY = "Arctic/Flux"

    def _pick_resolution(self, resolution_set: str, orientation: str, variant_index: int):
        candidates = RESOLUTION_SETS.get(resolution_set, RESOLUTIONS_1M)
        oriented = [
            (w, h) for (w, h) in candidates if _orientation(w, h) == orientation
        ]
        if not oriented:
            oriented = candidates

        idx = variant_index % len(oriented)
        return oriented[idx]

    @staticmethod
    def _parse_resolution_choice(choice: str):
        if choice == "auto_from_set":
            return None
        # Expect format like "Landscape: 1152x896" etc.
        try:
            label, dims = choice.split(":")
            dims = dims.strip()
            w, h = map(int, dims.split("x"))
            return (w, h)
        except Exception as exc:
            raise ValueError(f"Bad resolution format: {choice}") from exc

    def build(
        self, resolution_set, orientation, resolution_choice, variant_index, batch_size
    ):
        parsed = self._parse_resolution_choice(resolution_choice)
        if parsed is not None:
            target_w, target_h = parsed
        else:
            target_w, target_h = self._pick_resolution(
                resolution_set, orientation, variant_index
            )

        latent = torch.zeros(
            (batch_size, 4, target_h // 8, target_w // 8), dtype=torch.float32
        )

        return ({"samples": latent}, target_w, target_h)


class FluxPromptWithGuidance:
    """
    Combine CLIP text encode + FluxGuidance into a single node.
    Inputs: CLIP, prompt text, guidance value. Output: CONDITIONING ready for ksampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cinematic photo of a snow fox, 35mm, dusk light",
                    },
                ),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 200.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Arctic/Flux"

    def encode(self, clip, prompt, guidance):
        if comfy_nodes is None:
            raise RuntimeError("ComfyUI core nodes module 'nodes' not available.")

        text_node_cls = getattr(comfy_nodes, "CLIPTextEncode", None)
        guidance_cls = _resolve_flux_guidance_cls()

        if text_node_cls is None:
            raise RuntimeError(
                "Missing core node: CLIPTextEncode. Update ComfyUI to a Flux build."
            )
        if guidance_cls is None:
            raise RuntimeError(
                "Missing core node: FluxGuidance. Update ComfyUI to a Flux build or ensure flux extras are installed."
            )

        text_node = text_node_cls()
        base_conditioning = text_node.encode(clip, prompt)[0]

        guidance_node = guidance_cls()
        if hasattr(guidance_node, "apply_guidance"):
            guided = guidance_node.apply_guidance(base_conditioning, guidance)[0]
        elif hasattr(guidance_node, "encode"):
            guided = guidance_node.encode(base_conditioning, guidance)[0]
        else:
            fn_name = getattr(guidance_node, "FUNCTION", None)
            fn = getattr(guidance_node, fn_name, None) if fn_name else None
            if callable(fn):
                guided = fn(base_conditioning, guidance)[0]
            elif callable(guidance_node):
                guided = guidance_node(base_conditioning, guidance)[0]  # type: ignore
            else:
                raise RuntimeError(
                    "FluxGuidance does not expose apply_guidance/encode and is not callable. "
                    f"Found FUNCTION={fn_name!r}; available attrs: {dir(guidance_node)}"
                )

        return (guided,)
