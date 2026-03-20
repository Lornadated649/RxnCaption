"""
BIVP Annotator (Bounding-box Index Visual Prompt)
==================================================
Draws blue bounding boxes and reading-order numeric labels on top of each
image, producing "visual-prompt" images that are fed to the VL model.

Input
-----
- A directory of raw cropped reaction images.
- A directory of per-image YOLO detection JSON files
  (output of molyolo/predict.py, each file: list of {confidence, bbox_xyxy}).

Output
------
- A directory of annotated images (same filenames as input).

Usage
-----
    python rxncaption/annotate.py \
        --image_root_dir  /path/to/cropped_images \
        --det_json_root_dir /path/to/yolo_det_json \
        --middle_root_dir  /path/to/output_annotated \
        --confidence_threshold 0.5
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_index_label(draw: ImageDraw.ImageDraw, num: int,
                      xmin: float, ymin: float,
                      font: ImageFont.ImageFont, padding: int = 2) -> None:
    """Overlay a black-background numeric label at the top-left of a box."""
    bb = draw.textbbox((0, 0), str(num), font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]

    fill_x1, fill_y1 = xmin, ymin - 20
    fill_x2 = fill_x1 + tw + 2 * padding
    fill_y2 = fill_y1 + 1.8 * th
    draw.rectangle((fill_x1, fill_y1, fill_x2, fill_y2), fill="black")
    draw.text((fill_x1 + padding, fill_y1 + padding),
              str(num), fill="white", font=font)


# ---------------------------------------------------------------------------
# Reading-order sort
# ---------------------------------------------------------------------------

def _group_by_row(bboxes: list) -> list:
    rows, current = [], [bboxes[0]]
    for box in bboxes[1:]:
        h = current[0][3] - current[0][1]
        if current[0][1] - h / 2 <= box[1] <= current[0][1] + h / 2:
            current.append(box)
        else:
            rows.append(current)
            current = [box]
    rows.append(current)
    return rows


def reading_order(bboxes: list) -> list:
    """Return xyxy boxes sorted in top-to-bottom, left-to-right order."""
    sorted_boxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
    rows = _group_by_row(sorted_boxes)
    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b[0]))
    return result


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------

def _filter_by_confidence(records: list, threshold: float) -> list:
    """Keep only detections whose confidence >= threshold."""
    return [r["bbox_xyxy"] for r in records if r.get("confidence", 0) >= threshold]


def annotate_image(image_path: str, bboxes: list,
                   line_width: int = 4,
                   min_font_size: int = 24, font_step: int = 12) -> Image.Image:
    """
    Draw blue bounding boxes and reading-order numeric labels on *image_path*.

    Parameters
    ----------
    image_path    : path to the source image.
    bboxes        : list of [x1, y1, x2, y2] detections (already filtered).
                    Coordinates are in original image pixel space (no scaling).
    line_width    : width of the drawn rectangles in pixels.
    min_font_size : smallest allowed font size for numeric labels.
    font_step     : increment between font-size tiers.

    Returns
    -------
    A PIL Image with boxes and labels drawn.
    """
    image = Image.open(image_path).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    if not bboxes:
        return annotated

    # Sort into reading order before assigning indices
    ordered = reading_order(bboxes)

    # Dynamic font sizing based on smallest box dimension
    min_dim = min(min(b[2] - b[0], b[3] - b[1]) for b in ordered)
    if min_dim < 50:
        font_size = max(int(min_dim * 0.5), min_font_size)
    elif min_dim < 100:
        font_size = max(int(min_dim * 0.4), min_font_size + font_step)
    else:
        font_size = max(int(min_dim * 0.3), min_font_size + 2 * font_step)
    font_size = min(font_size, 48)
    font = ImageFont.load_default(size=font_size)

    for i, (x1, y1, x2, y2) in enumerate(ordered):
        draw.rectangle((x1, y1, x2, y2), outline="blue", width=line_width)
        _draw_index_label(draw, i + 1, x1, y1 + 20, font)

    return annotated


def process_directory(args: argparse.Namespace) -> None:
    """Annotate all images found in args.image_root_dir."""
    out_dir = os.path.join(args.middle_root_dir,
                           f"threshold_{args.confidence_threshold}")
    os.makedirs(out_dir, exist_ok=True)

    image_files = os.listdir(args.image_root_dir)
    stats = {"total": len(image_files), "new": 0, "skipped": 0, "no_json": 0}

    for fname in image_files:
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            stats["skipped"] += 1
            continue

        image_path = os.path.join(args.image_root_dir, fname)

        # Find the corresponding detection JSON
        json_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(
                args.det_json_root_dir,
                fname.replace(ext, ".json"),
            )
            if os.path.exists(candidate):
                json_path = candidate
                break

        if json_path is None:
            stats["no_json"] += 1
            print(f"[WARN] No detection JSON for {fname}, skipping.")
            continue

        with open(json_path) as f:
            records = json.load(f)

        bboxes = _filter_by_confidence(records, args.confidence_threshold)
        annotated = annotate_image(
            image_path, bboxes,
            line_width=args.line_width,
            min_font_size=args.min_font_size,
            font_step=args.font_step,
        )
        annotated.save(out_path)
        stats["new"] += 1
        print(f"[OK] {fname}")

    print("\n" + "=" * 50)
    print(f"Total images : {stats['total']}")
    print(f"Newly created: {stats['new']}")
    print(f"Skipped      : {stats['skipped']}")
    print(f"Missing JSON : {stats['no_json']}")
    print(f"Output dir   : {out_dir}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BIVP: annotate images with bounding boxes and numeric indices."
    )
    parser.add_argument(
        "--image_root_dir", required=True,
        help="Directory containing raw cropped images.",
    )
    parser.add_argument(
        "--det_json_root_dir", required=True,
        help="Directory containing per-image YOLO detection JSON files.",
    )
    parser.add_argument(
        "--middle_root_dir", required=True,
        help="Root output directory; annotated images are written to "
             "<middle_root_dir>/threshold_<conf>/.",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5,
        help="Minimum YOLO confidence to include a box (default: 0.5).",
    )
    parser.add_argument(
        "--line_width", type=int, default=4,
        help="Box outline width in pixels (default: 4).",
    )
    parser.add_argument(
        "--min_font_size", type=int, default=24,
        help="Minimum label font size (default: 24).",
    )
    parser.add_argument(
        "--font_step", type=int, default=12,
        help="Font size increment per box-size tier (default: 12).",
    )
    args = parser.parse_args()
    process_directory(args)


if __name__ == "__main__":
    main()
