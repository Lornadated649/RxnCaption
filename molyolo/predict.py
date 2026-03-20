"""
MolYOLO Inference Module
========================
Runs YOLOv10-based molecular structure detection on a directory of images.

Outputs per-image JSON files (bbox_xyxy + confidence) and optional
visualizations (annotated images, visual-prompt images with index labels).

Usage
-----
    python molyolo/predict.py \
        --img_dir  /path/to/images \
        --weights  molyolo/weights/MolYOLO.pt \
        --output_dir ./outputs \
        --output_name run_01 \
        --gpu_num 1 \
        --visual_prompt
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# YOLOv10 from the bundled ultralytics fork
from ultralytics import YOLOv10

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_index_label(draw: ImageDraw.ImageDraw, num: int,
                      xmin: float, ymin: float,
                      font: ImageFont.ImageFont, padding: int = 2) -> None:
    """Draw a black-background numeric label at (xmin, ymin)."""
    bbox = draw.textbbox((0, 0), str(num), font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    fill_x1 = xmin
    fill_y1 = ymin - 20
    fill_x2 = fill_x1 + text_w + 2 * padding
    fill_y2 = fill_y1 + 1.8 * text_h
    draw.rectangle((fill_x1, fill_y1, fill_x2, fill_y2), fill="black")
    draw.text((fill_x1 + padding, fill_y1 + padding), str(num),
              fill="white", font=font)


# ---------------------------------------------------------------------------
# Reading-order sorting
# ---------------------------------------------------------------------------

def _group_by_row(bboxes: list) -> list:
    """Group xyxy boxes into rows based on vertical overlap."""
    rows = []
    if not bboxes:
        return rows
    current_row = [bboxes[0]]
    for box in bboxes[1:]:
        ref = current_row[0]
        h = ref[3] - ref[1]
        if ref[1] - h / 2 <= box[1] <= ref[1] + h / 2:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)
    return rows


def reading_order(bboxes: list) -> list:
    """Sort xyxy boxes in top-to-bottom, left-to-right reading order."""
    sorted_boxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
    rows = _group_by_row(sorted_boxes)
    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b[0]))
    return result


# ---------------------------------------------------------------------------
# NMS / overlap suppression
# ---------------------------------------------------------------------------

def suppress_overlapping(bboxes: list, confs: list = None,
                          iou_thresh: float = 0.5) -> np.ndarray:
    """
    Greedy overlap suppression.
    Removes boxes with IoU > iou_thresh OR overlap-ratio > 0.7 with a
    higher-confidence box.

    Returns a boolean keep-mask (same length as bboxes).
    """
    bboxes = np.asarray(bboxes)
    n = len(bboxes)
    if n == 0:
        return np.array([], dtype=bool)
    confs = np.ones(n) if confs is None else np.asarray(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    keep = np.ones(n, dtype=bool)

    for i in np.argsort(confs)[::-1]:
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            ix1 = max(bboxes[i, 0], bboxes[j, 0])
            iy1 = max(bboxes[i, 1], bboxes[j, 1])
            ix2 = min(bboxes[i, 2], bboxes[j, 2])
            iy2 = min(bboxes[i, 3], bboxes[j, 3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = areas[i] + areas[j] - inter
            iou = inter / (union + 1e-6)
            overlap_ratio = inter / (min(areas[i], areas[j]) + 1e-6)
            if iou > iou_thresh or overlap_ratio > 0.7:
                if confs[i] >= confs[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return keep


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _save_json(image_path: str, output_dir: str, run_name: str,
               result) -> None:
    """Save detection result as JSON: list of {class_name, confidence, bbox_xyxy}."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, run_name, "json", stem + ".json")
    boxes = result.boxes.xyxy.int().tolist()
    confs = result.boxes.conf.tolist()
    cls_ids = result.boxes.cls.int().tolist()
    records = [
        {"class_name": result.names[c], "confidence": cf, "bbox_xyxy": b}
        for b, cf, c in zip(boxes, confs, cls_ids)
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, indent=4)
    print(f"[JSON] {os.path.basename(out_path)}")


def _save_visual_image(image_path: str, output_dir: str, run_name: str,
                       result, save_original: bool = False) -> None:
    """Save YOLO-annotated image (and optionally the original)."""
    fname = os.path.basename(image_path)
    vis_path = os.path.join(output_dir, run_name, "visual_images", fname)
    vis = Image.fromarray(result.plot(line_width=2, font_size=10)[..., ::-1])
    vis.save(vis_path)
    print(f"[VIS] {fname}")
    if save_original:
        ori_path = os.path.join(output_dir, run_name, "origin_images", fname)
        Image.open(image_path).save(ori_path)
        print(f"[ORI] {fname}")


def _save_vp_image(image_path: str, output_dir: str, run_name: str,
                   result) -> None:
    """
    Save a visual-prompt image: blue bounding boxes + reading-order numeric
    labels overlaid on the original image.
    """
    fname = os.path.basename(image_path)
    out_path = os.path.join(output_dir, run_name, "vp_image", fname)

    boxes = result.boxes.xyxy.int().tolist()
    image = Image.open(image_path)

    if not boxes:
        image.save(out_path)
        print(f"[VP] {fname} (no detections)")
        return

    boxes = reading_order(boxes)

    # Dynamic font size based on smallest box dimension
    min_dim = min(min(b[2] - b[0], b[3] - b[1]) for b in boxes)
    if min_dim < 50:
        font_size = max(int(min_dim * 0.5), 8)
    elif min_dim < 100:
        font_size = max(int(min_dim * 0.4), 10)
    else:
        font_size = max(int(min_dim * 0.3), 12)
    font_size = min(font_size, 20)
    font = ImageFont.load_default(size=font_size)

    vp = image.copy()
    draw = ImageDraw.Draw(vp)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle((x1, y1, x2, y2), outline="blue", width=2)
        _draw_index_label(draw, i + 1, x1, y1 + 20, font)

    vp.save(out_path)
    print(f"[VP] {fname}")


# ---------------------------------------------------------------------------
# Per-process model (for multiprocessing)
# ---------------------------------------------------------------------------

_model = None  # module-level, one per worker process


def _init_model(weights: str, gpu_queue: mp.Queue, gpu_num: int) -> None:
    """Process-pool initializer: load the YOLOv10 model onto an assigned GPU."""
    global _model
    try:
        gpu_id = gpu_queue.get() % gpu_num
    except Exception:
        gpu_id = 0
    print(f"[Worker] Model loaded on cuda:{gpu_id}")
    _model = YOLOv10(weights).to(f"cuda:{gpu_id}")


def _process_one(image_path: str, args: argparse.Namespace) -> None:
    """Run inference + save outputs for a single image (called in worker)."""
    result = _model.predict(
        image_path,
        imgsz=args.image_size,
        conf=args.conf,
        iou=args.iou_thresh,
        project=args.output_dir,
        name=args.output_name,
        save_txt=False,
        exist_ok=True,
    )[0]

    # Apply custom overlap suppression
    boxes = result.boxes.xyxy.int().tolist()
    confs = result.boxes.conf.tolist()
    if boxes:
        keep = suppress_overlapping(boxes, confs, args.iou_thresh)
        result.boxes.data = result.boxes.data[keep]

    _save_json(image_path, args.output_dir, args.output_name, result)

    if args.visual_prompt:
        _save_vp_image(image_path, args.output_dir, args.output_name, result)

    if args.visual_image:
        _save_visual_image(image_path, args.output_dir, args.output_name,
                           result, args.save_origin_image)


# ---------------------------------------------------------------------------
# Directory setup helpers
# ---------------------------------------------------------------------------

def _make_output_dirs(args: argparse.Namespace) -> None:
    base = os.path.join(args.output_dir, args.output_name)
    os.makedirs(os.path.join(base, "json"), exist_ok=True)
    if args.save_txt:
        os.makedirs(os.path.join(base, "labels"), exist_ok=True)
    if args.save_origin_image:
        os.makedirs(os.path.join(base, "origin_images"), exist_ok=True)
    if args.visual_image:
        os.makedirs(os.path.join(base, "visual_images"), exist_ok=True)
    if args.visual_prompt:
        os.makedirs(os.path.join(base, "vp_image"), exist_ok=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MolYOLO: molecular structure detection inference"
    )
    parser.add_argument("--img_dir", required=True,
                        help="Directory containing input images.")
    parser.add_argument("--weights", default="molyolo/weights/MolYOLO.pt",
                        help="Path to the YOLOv10 checkpoint.")
    parser.add_argument("--output_dir", default="outputs",
                        help="Root directory for inference outputs.")
    parser.add_argument("--output_name", default="run",
                        help="Sub-folder name for this run.")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Inference image size (pixels, square).")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for detections.")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU threshold for NMS.")
    parser.add_argument("--gpu_num", type=int, default=1,
                        help="Number of GPUs available.")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--visual_prompt", action="store_true",
                        help="Save visual-prompt images (boxes + index labels).")
    parser.add_argument("--visual_image", action="store_true",
                        help="Save YOLO-annotated images.")
    parser.add_argument("--save_origin_image", action="store_true",
                        help="Copy original images to output dir.")
    parser.add_argument("--save_txt", action="store_true",
                        help="Save detections as YOLO-format .txt files.")
    args = parser.parse_args()

    _make_output_dirs(args)

    # Collect images
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    image_paths = [
        os.path.join(args.img_dir, f)
        for f in os.listdir(args.img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]
    print(f"Found {len(image_paths)} images in {args.img_dir}")

    # Build GPU queue for worker assignment
    gpu_queue: mp.Queue = mp.Queue()
    for i in range(args.n_workers):
        gpu_queue.put(i)

    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_model,
        initargs=(args.weights, gpu_queue, args.gpu_num),
    ) as pool:
        futures = {pool.submit(_process_one, p, args): p for p in image_paths}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print(f"[ERROR] {futures[fut]}: {exc}")

    print("All images processed.")


if __name__ == "__main__":
    main()
