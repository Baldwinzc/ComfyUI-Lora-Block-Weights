"""Assemble sweep output images into a labeled grid.

The Batch sweep node saves images in row-major order (outer loop = blocks,
inner loop = values), so given the first file in the sequence and the
block/value lists used, this rebuilds the grid.

Usage:
    python make_grid.py <first_image_path> \\
        --blocks D00,D01,...,D18 \\
        --values 0,0.5,1.0 \\
        --output grid.png

Optional:
    --thumb 256        Resize each cell to <pixels> on the long side
    --label-size 24    Label font pixel size
    --pad 8            Padding between cells
"""

import argparse
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont


_NUM_RE = re.compile(r"_(\d+)_")


def _next_path(first_path: str, offset: int) -> str:
    """Increment the numeric id in the ComfyUI default filename pattern."""
    directory, name = os.path.split(first_path)
    m = _NUM_RE.search(name)
    if not m:
        raise ValueError(
            f"Filename '{name}' does not match ComfyUI default pattern "
            "(...._00001_.png). Pass the sweep's first file explicitly."
        )
    base_id = int(m.group(1))
    width = len(m.group(1))
    new_num = f"{base_id + offset:0{width}d}"
    new_name = name[:m.start(1)] + new_num + name[m.end(1):]
    return os.path.join(directory, new_name)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def build_grid(first_path: str, blocks: list, values: list,
               thumb: int = 256, label_size: int = 24,
               pad: int = 8) -> Image.Image:
    rows = len(blocks)
    cols = len(values)
    expected = rows * cols

    cells = []
    for i in range(expected):
        path = _next_path(first_path, i)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing image #{i + 1}/{expected}: {path}"
            )
        cells.append(_resize(Image.open(path).convert("RGB"), thumb))

    cell_w = max(c.width for c in cells)
    cell_h = max(c.height for c in cells)
    label_h = label_size + 6
    cell_full_h = cell_h + label_h

    row_label_w = label_size * 4
    col_label_h = label_size + 6

    grid_w = row_label_w + cols * cell_w + (cols + 1) * pad
    grid_h = col_label_h + rows * cell_full_h + (rows + 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    font = _load_font(label_size)

    for col_idx, val in enumerate(values):
        x = row_label_w + pad + col_idx * (cell_w + pad)
        text = f"v={val}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (cell_w - text_w) // 2, 2), text,
                  fill="black", font=font)

    for row_idx, block in enumerate(blocks):
        y = col_label_h + pad + row_idx * (cell_full_h + pad)
        bbox = draw.textbbox((0, 0), block, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text((4, y + (cell_h - text_h) // 2), block,
                  fill="black", font=font)

        for col_idx in range(cols):
            cell = cells[row_idx * cols + col_idx]
            x = row_label_w + pad + col_idx * (cell_w + pad)
            grid.paste(cell, (x, y))
            tag = f"{block} v={values[col_idx]}"
            draw.text((x + 4, y + cell_h + 2), tag,
                      fill="black", font=font)

    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_image",
                        help="Path to the first sweep image, e.g. "
                             "/data/ComfyUI/output/ComfyUI_00001_.png")
    parser.add_argument("--blocks", required=True,
                        help="Comma-separated block tags in the sweep order")
    parser.add_argument("--values", required=True,
                        help="Comma-separated values in the sweep order")
    parser.add_argument("--output", default="grid.png",
                        help="Output path (default: grid.png)")
    parser.add_argument("--thumb", type=int, default=256,
                        help="Max side per cell in pixels (default 256, "
                             "0 = original size)")
    parser.add_argument("--label-size", type=int, default=24,
                        help="Label font size (default 24)")
    parser.add_argument("--pad", type=int, default=8,
                        help="Padding between cells (default 8)")
    args = parser.parse_args()

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    values = [v.strip() for v in args.values.split(",") if v.strip()]

    grid = build_grid(args.first_image, blocks, values,
                      thumb=args.thumb,
                      label_size=args.label_size,
                      pad=args.pad)
    grid.save(args.output)
    print(f"Saved {grid.width}x{grid.height} grid to {args.output} "
          f"({len(blocks)} rows x {len(values)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
