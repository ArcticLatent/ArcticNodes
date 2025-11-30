import importlib

import torch
import torch.nn.functional as F

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

    # Common module names where FluxGuidance may live depending on ComfyUI build.
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
    """
    Resize an IMAGE to a Flux-safe resolution with orientation and upscale control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_set": (
                    ["1M_flux", "1_5M_flux", "2M_flux"],
                    {"default": "1M_flux"},
                ),
                "orientation_mode": (
                    ["auto", "landscape", "portrait", "square"],
                    {"default": "auto"},
                ),
                "allow_upscale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "Allow Upscale (otherwise only downscale)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "Arctic/Flux"

    def _pick_best_resolution(
        self,
        width: int,
        height: int,
        resolution_set: str,
        orientation_mode: str,
        allow_upscale: bool,
    ):
        src_aspect = width / height if height != 0 else 1.0
        src_pixels = width * height

        candidates = RESOLUTION_SETS.get(resolution_set, RESOLUTIONS_1M)

        if orientation_mode == "auto":
            desired_orientation = _orientation(width, height)
        else:
            desired_orientation = orientation_mode

        oriented_candidates = [
            (w_tgt, h_tgt)
            for (w_tgt, h_tgt) in candidates
            if _orientation(w_tgt, h_tgt) == desired_orientation
        ]

        if not oriented_candidates:
            oriented_candidates = candidates

        best_res = None
        best_score = None

        for w_tgt, h_tgt in oriented_candidates:
            if not allow_upscale and (w_tgt > width or h_tgt > height):
                continue

            tgt_aspect = w_tgt / h_tgt if h_tgt != 0 else 1.0
            tgt_pixels = w_tgt * h_tgt

            aspect_diff = abs(tgt_aspect - src_aspect)
            pixel_diff_ratio = abs(tgt_pixels - src_pixels) / max(src_pixels, 1)
            score = aspect_diff + 0.25 * pixel_diff_ratio

            if best_score is None or score < best_score:
                best_score = score
                best_res = (w_tgt, h_tgt)

        if best_res is None:
            for w_tgt, h_tgt in oriented_candidates:
                tgt_aspect = w_tgt / h_tgt if h_tgt != 0 else 1.0
                aspect_diff = abs(tgt_aspect - src_aspect)
                if best_score is None or aspect_diff < best_score:
                    best_score = aspect_diff
                    best_res = (w_tgt, h_tgt)

        return best_res

    def resize(self, image, resolution_set, orientation_mode, allow_upscale):
        """
        image: torch.Tensor [B, H, W, C] in 0–1 range.
        """
        if not torch.is_tensor(image):
            raise ValueError("Expected IMAGE tensor")

        b, h, w, c = image.shape

        target_w, target_h = self._pick_best_resolution(
            w, h, resolution_set, orientation_mode, allow_upscale
        )

        if (w, h) == (target_w, target_h):
            return (image, target_w, target_h)

        x = image.movedim(-1, 1)  # [B, C, H, W]

        x_resized = F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        out = x_resized.movedim(1, -1)  # back to [B, H, W, C]
        out = torch.clamp(out, 0.0, 1.0)

        return (out, target_w, target_h)


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
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
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
        # FluxGuidance uses FUNCTION "apply_guidance" in mainline; fall back to common alternatives.
        if hasattr(guidance_node, "apply_guidance"):
            guided = guidance_node.apply_guidance(base_conditioning, guidance)[0]
        elif hasattr(guidance_node, "encode"):
            guided = guidance_node.encode(base_conditioning, guidance)[0]
        else:
            # Last-resort call style for unexpected implementations.
            guided = guidance_node(base_conditioning, guidance)[0]  # type: ignore

        return (guided,)
