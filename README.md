# ComfyUI-LoraBlockSweep

Per-block LoRA weighting for Flux, designed for systematic XY-plot experiments.

## Why

Bobs LoRA Loader exposes 14 conceptual block groups; this exposes Flux's actual
57 transformer blocks (19 double + 38 single) so a single-block sweep gives a
clean signal per layer. Input/output layers (img_in / txt_in / time_in /
vector_in / guidance_in / final_layer) follow `baseline_weight` and are not
sweep targets.

## Install

Copy or symlink this folder into `<ComfyUI>/custom_nodes/`:

    cp -r D:/code/ComfyUI-LoraBlockSweep <ComfyUI>/custom_nodes/

Restart ComfyUI.

## Nodes

### LoRA Block Sweep (FLUX)

Drop-in replacement for `LoraLoader`. Takes one `target_block` (D00..D18 or
S00..S37) and one `target_value`; every other block is set to
`baseline_weight`.

- `baseline_weight = 1.0` -> Knock-out experiment (others stay full, target varies)
- `baseline_weight = 0.0` -> Solo experiment (others off, target alone)

Pair with Efficiency Nodes XY Plot:

- X axis: `XY Input: String` with the 57 block names, comma separated
- Y axis: `XY Input: Number` with `0,0.25,0.5,0.75,1.0`

Result: 57 x 5 = 285 image grid.

### LoRA Block Sweep Custom (FLUX)

After the sweep narrows things down, use this node to set every block
individually via a 57-value comma list (order: D00..D18, S00..S37).
