"""Quickly inspect a LoRA file to see how its weights are distributed across
CLIP / UNet / other key buckets, plus a sample of actual key names.

Usage:
    python3 inspect_lora.py /path/to/lora.safetensors
"""

import sys
from safetensors import safe_open


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_lora.py <path-to-lora.safetensors>")
        sys.exit(1)

    path = sys.argv[1]
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())

    clip_keys = [k for k in keys
                 if "te_" in k or "text_model" in k or "lora_te" in k
                 or "_t5_" in k]
    unet_keys = [k for k in keys
                 if "double_blocks" in k or "single_blocks" in k]
    other = [k for k in keys
             if k not in clip_keys and k not in unet_keys]

    print("File:", path)
    print("Total keys:        ", len(keys))
    print("CLIP-side:         ", len(clip_keys))
    print("UNet (D/S blocks): ", len(unet_keys))
    print("Other:             ", len(other))

    print("\nFirst 10 keys:")
    for k in keys[:10]:
        print("  " + k)

    if other:
        print("\nFirst 10 'other' keys (input/output layers, etc.):")
        for k in other[:10]:
            print("  " + k)


if __name__ == "__main__":
    main()
