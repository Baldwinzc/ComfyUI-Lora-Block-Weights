import re
from collections import defaultdict

import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.utils


DOUBLE_BLOCK_COUNT = 19
SINGLE_BLOCK_COUNT = 38

DOUBLE_BLOCK_NAMES = [f"D{i:02d}" for i in range(DOUBLE_BLOCK_COUNT)]
SINGLE_BLOCK_NAMES = [f"S{i:02d}" for i in range(SINGLE_BLOCK_COUNT)]
ALL_TARGET_BLOCKS = DOUBLE_BLOCK_NAMES + SINGLE_BLOCK_NAMES

_RE_DOUBLE = re.compile(r"diffusion_model\.double_blocks\.(\d+)\.")
_RE_SINGLE = re.compile(r"diffusion_model\.single_blocks\.(\d+)\.")


def _classify_key(key: str) -> str:
    """Return block tag for a state_dict key.

    Returns one of: 'D00'..'D18', 'S00'..'S37', or 'extras'.
    'extras' covers img_in / txt_in / time_in / vector_in / guidance_in / final_layer.
    """
    m = _RE_DOUBLE.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < DOUBLE_BLOCK_COUNT:
            return f"D{idx:02d}"
    m = _RE_SINGLE.search(key)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < SINGLE_BLOCK_COUNT:
            return f"S{idx:02d}"
    return "extras"


def _build_block_strengths(target_block: str, target_value: float,
                           baseline_weight: float) -> dict:
    """Build {block_tag: strength} map.

    target_block is set to target_value, every other tag is baseline_weight.
    'extras' (input/output layers) also follows baseline_weight so the LoRA's
    embedding-side modifications stay consistent with the experiment baseline.
    """
    strengths = {tag: baseline_weight for tag in ALL_TARGET_BLOCKS}
    strengths["extras"] = baseline_weight
    if target_block in strengths:
        strengths[target_block] = target_value
    return strengths


def _apply_blockwise_patches(model_patcher, loaded_patches: dict,
                             block_strengths: dict):
    """Group loaded patches by block tag, then call add_patches once per group
    with the per-block strength. Empty-strength groups are skipped to avoid
    no-op work.
    """
    by_block = defaultdict(dict)
    for k, v in loaded_patches.items():
        tag = _classify_key(k)
        by_block[tag][k] = v

    applied_keys = set()
    for tag, patches in by_block.items():
        if not patches:
            continue
        strength = block_strengths.get(tag, 0.0)
        keys = model_patcher.add_patches(patches, strength)
        applied_keys.update(keys)
    return applied_keys


class LoraBlockSweepFlux:
    """Single-block sweep for Flux LoRA.

    Pair with Efficiency Nodes XY Plot:
      X axis -> XY Input: String, comma-list of block tags (e.g. D00,D01,...,S37)
      Y axis -> XY Input: Number, sweep values (e.g. 0,0.25,0.5,0.75,1.0)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "target_block": ("STRING",
                                 {"default": ALL_TARGET_BLOCKS[0],
                                  "tooltip": "Block tag: D00..D18 or S00..S37. "
                                             "Empty / unknown -> no target, all blocks at baseline_weight."}),
                "target_value": ("FLOAT",
                                 {"default": 1.0, "min": 0.0, "max": 2.0,
                                  "step": 0.05}),
                "baseline_weight": ("FLOAT",
                                    {"default": 1.0, "min": 0.0, "max": 2.0,
                                     "step": 0.05,
                                     "tooltip": "Knock-out: 1.0 (others stay full, target varies). "
                                                "Solo: 0.0 (others off, only target varies)."}),
                "clip_strength": ("FLOAT",
                                  {"default": 1.0, "min": -2.0, "max": 2.0,
                                   "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "apply"
    CATEGORY = "LoraBlockSweep"

    def apply(self, model, clip, lora_name, target_block, target_value,
              baseline_weight, clip_strength):
        target_block = (target_block or "").strip().upper()
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        key_map = {}
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        lora_sd = comfy.lora_convert.convert_lora(lora_sd)
        loaded = comfy.lora.load_lora(lora_sd, key_map)

        block_strengths = _build_block_strengths(
            target_block, target_value, baseline_weight)

        new_model = model.clone()
        applied = _apply_blockwise_patches(new_model, loaded, block_strengths)

        new_clip = clip.clone()
        clip_applied = new_clip.add_patches(loaded, clip_strength)
        applied.update(clip_applied)

        not_loaded = [k for k in loaded if k not in applied]
        info = (f"target={target_block}={target_value:.3f} "
                f"baseline={baseline_weight:.3f} "
                f"clip={clip_strength:.3f} "
                f"patched={len(applied)} skipped={len(not_loaded)}")
        if not_loaded:
            info += f" first_skip={not_loaded[0]}"
        return (new_model, new_clip, info)


class LoraBlockSweepFluxCustom:
    """Manual per-block weights for fine tuning after the sweep narrows things down.

    Accepts a 57-comma-separated weight string in the order:
      D00,D01,...,D18,S00,S01,...,S37
    Anything missing or non-numeric falls back to baseline_weight.
    """

    @classmethod
    def INPUT_TYPES(cls):
        default_weights = ",".join(["1.0"] * len(ALL_TARGET_BLOCKS))
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "weights": ("STRING",
                            {"default": default_weights, "multiline": True}),
                "baseline_weight": ("FLOAT",
                                    {"default": 1.0, "min": 0.0, "max": 2.0,
                                     "step": 0.05}),
                "clip_strength": ("FLOAT",
                                  {"default": 1.0, "min": -2.0, "max": 2.0,
                                   "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "apply"
    CATEGORY = "LoraBlockSweep"

    def apply(self, model, clip, lora_name, weights, baseline_weight,
              clip_strength):
        parts = [w.strip() for w in weights.split(",")]
        per_block = {}
        for i, tag in enumerate(ALL_TARGET_BLOCKS):
            if i < len(parts):
                try:
                    per_block[tag] = float(parts[i])
                    continue
                except ValueError:
                    pass
            per_block[tag] = baseline_weight
        per_block["extras"] = baseline_weight

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        key_map = {}
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        lora_sd = comfy.lora_convert.convert_lora(lora_sd)
        loaded = comfy.lora.load_lora(lora_sd, key_map)

        new_model = model.clone()
        applied = _apply_blockwise_patches(new_model, loaded, per_block)

        new_clip = clip.clone()
        clip_applied = new_clip.add_patches(loaded, clip_strength)
        applied.update(clip_applied)

        info = (f"custom weights, baseline={baseline_weight:.3f} "
                f"clip={clip_strength:.3f} patched={len(applied)}")
        return (new_model, new_clip, info)


NODE_CLASS_MAPPINGS = {
    "LoraBlockSweepFlux": LoraBlockSweepFlux,
    "LoraBlockSweepFluxCustom": LoraBlockSweepFluxCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraBlockSweepFlux": "LoRA Block Sweep (FLUX)",
    "LoraBlockSweepFluxCustom": "LoRA Block Sweep Custom (FLUX)",
}
