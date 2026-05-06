# Usage Guide

## Install

1. Find your ComfyUI install (the one with `custom_nodes/` folder)
2. Copy this whole folder there:

   ```
   cp -r D:/code/ComfyUI-LoraBlockSweep <ComfyUI>/custom_nodes/
   ```

3. Restart ComfyUI

## Replace LoraLoader 78 in your workflow

In the Z-Image -> Flux Kontext workflow, node 78 (`LoraLoader`) is the one to
replace. Steps:

1. Right-click node 78 -> Remove (note its connections first):

   ```
   model in  <- UNETLoader 73
   clip  in  <- DualCLIPLoader 74
   MODEL out -> KSampler 83
   CLIP  out -> CLIPTextEncode 86
   ```

2. Add `LoRA Block Sweep (FLUX)` node, reconnect the same way

3. Set `lora_name = xdt_i2i/xdt_char_i2i_v1_copy.safetensors`

## Knock-out experiment (recommended first round)

- `baseline_weight = 1.0`
- `target_block` = (will be swept by XY)
- `target_value` = (will be swept by XY)
- `clip_strength = 1.0`

Add Efficiency Nodes XY Plot:

- X axis: `XY Input: String`
  - Override input: `target_block`
  - Values (paste this comma-separated string):

    ```
    D00,D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13,D14,D15,D16,D17,D18,S00,S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37
    ```

- Y axis: `XY Input: Number`
  - Override input: `target_value`
  - Values: `0,0.25,0.5,0.75,1.0`

Result: 57 x 5 = 285 image grid. Each cell shows what happens when ONE block
is dialed to that strength while the other 56 stay at 1.0.

Reading the grid:

- Column-wise nearly identical -> that block is non-critical, can be lowered
- Column-wise dramatic change -> that block is critical, keep at 1.0
- Column-wise improving as value drops -> that block has unwanted effects,
  consider lowering it permanently

## Solo experiment (optional second round)

Same setup, but change `baseline_weight = 0.0`. Now each cell shows what the
block alone contributes (everything else off). Useful for understanding
what each block "knows."

## Custom fine-tune (after the sweep)

Use `LoRA Block Sweep Custom (FLUX)`. Paste a 57-value string in the order:

    D00,D01,...,D18,S00,S01,...,S37

Example: keep all double blocks, drop late single blocks:

    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0

## Tips

- Set seed to fixed in KSampler so the only variable in the grid is block weight
- Use a low resolution (e.g. 768x768) for the sweep to save time, then re-test
  the best settings at full resolution
- 285 images at 25 steps will take 1-3 hours on a single GPU. Consider doing
  just the double blocks first (19 x 5 = 95 images) before committing
- The `info` STRING output can be wired to a `ShowText` node so each cell
  displays its parameters
