"""
Generate YOLO-to-GT Mapping Dictionary
=======================================
For each image, matches each YOLO-detected bounding box to the closest
ground-truth bbox (by IoU). The resulting mapping is used downstream
(transform_yolo_detections.py, transform_prediction_to_gtformat.py) to
translate YOLO reading-order indices back into the GT coordinate system.

Output JSON: a list where each entry is
    {
      "file_name": "<original GT file name>",
      "map_dict": {
          "1": {"gt_id": <int or -1>, "yolo_bbox": [x,y,w,h], "gt_bbox": [x,y,w,h]},
          "2": {...},
          ...
      }
    }

``gt_id = -1`` means no GT box achieved IoU >= 0.5.

Usage
-----
    python tools/generate_mapdict.py \
        --raw_gt_path  data/ground_truth.json \
        --yolo_path    data/det_json/ \
        --yolo_map_dict data/mapdict_from_yolo_to_gt.json \
        --dpi 400
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Reading-order sort (xyxy boxes)
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
    """Sort xyxy boxes top-to-bottom, left-to-right."""
    sorted_boxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
    rows = _group_by_row(sorted_boxes)
    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b[0]))
    return result


# ---------------------------------------------------------------------------
# IoU (xyxy)
# ---------------------------------------------------------------------------

def _iou_xyxy(a: list, b: list) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a YOLO-index → GT-id mapping dictionary."
    )
    parser.add_argument(
        "--raw_gt_path", required=True,
        help="Path to the raw ground-truth JSON (COCO-like, with 'images' list).",
    )
    parser.add_argument(
        "--yolo_path", required=True,
        help="Directory containing per-image YOLO detection JSON files "
             "(output of molyolo/predict.py).",
    )
    parser.add_argument(
        "--yolo_map_dict", required=True,
        help="Path for the output mapping JSON file.",
    )
    parser.add_argument(
        "--dpi", type=int, default=400,
        help="DPI of input images. YOLO box coordinates are scaled by "
             "(dpi / 200) before matching. Default: 400.",
    )
    args = parser.parse_args()

    with open(args.raw_gt_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    result_list = []
    processed, not_found = 0, 0

    for image_data in raw_data["images"]:
        raw_fname = image_data["file_name"]
        # Strip sub-folder prefix (e.g. "single/foo.jpg" → "foo.json")
        basename = os.path.basename(raw_fname)
        json_fname = os.path.splitext(basename)[0] + ".json"
        yolo_json_path = os.path.join(args.yolo_path, json_fname)

        if not os.path.exists(yolo_json_path):
            not_found += 1
            print(f"[WARN] YOLO file not found: {yolo_json_path}")
            continue

        with open(yolo_json_path, encoding="utf-8") as f:
            yolo_data = json.load(f)

        scale = args.dpi / 200.0
        yolo_boxes_xyxy = [
            [c * scale for c in elem["bbox_xyxy"]]
            for elem in yolo_data
        ]

        if not yolo_boxes_xyxy:
            result_list.append({"file_name": raw_fname, "map_dict": {}})
            processed += 1
            continue

        # Sort into reading order and assign 1-based indices
        ordered_boxes = reading_order(yolo_boxes_xyxy)
        gt_items = image_data.get("bboxes", [])

        map_dict: dict = {}
        for idx, y_box in enumerate(ordered_boxes, start=1):
            best_gt_id, best_gt_bbox, best_iou = -1, [0, 0, 0, 0], 0.0
            for gt in gt_items:
                x, y, w, h = gt["bbox"]
                gt_xyxy = [x, y, x + w, y + h]
                iou = _iou_xyxy(y_box, gt_xyxy)
                if iou > best_iou:
                    best_iou, best_gt_id, best_gt_bbox = iou, gt["id"], gt["bbox"]

            if best_iou < 0.5:
                best_gt_id, best_gt_bbox = -1, [0, 0, 0, 0]

            x1, y1, x2, y2 = y_box
            yolo_bbox_wh = [x1, y1, x2 - x1, y2 - y1]
            map_dict[idx] = {
                "gt_id":      best_gt_id,
                "yolo_bbox":  yolo_bbox_wh,
                "gt_bbox":    best_gt_bbox,
            }

        result_list.append({"file_name": raw_fname, "map_dict": map_dict})
        processed += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.yolo_map_dict)), exist_ok=True)
    with open(args.yolo_map_dict, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Processed: {processed}, Not found: {not_found}")
    print(f"Output: {args.yolo_map_dict}")


if __name__ == "__main__":
    main()
