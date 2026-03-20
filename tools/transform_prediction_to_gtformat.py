"""
Transform Model Predictions → Evaluation GT Format
===================================================
Converts the raw prediction JSON (output of transform_jsonl_to_json.py)
into the COCO-like JSON format expected by rxncaption/evaluate.py.

Two variants are supported via ``--mode``:

* **trained**  – for predictions from the fine-tuned RxnCaption-VL model.
  Uses the YOLO→GT mapping file to translate reading-order indices back
  into GT coordinate space.

* **zero_shot** – for zero-shot predictions from a baseline model.
  The model may output text/identifier role members; these are treated as
  new virtual bboxes with empty coordinates.

In both cases the output schema matches the ground-truth format::

    {
      "images": [
        {
          "file_name": "foo.png",
          "bboxes": [...],
          "reactions": [
            {"reactants": [0], "conditions": [1], "products": [2]}
          ]
        }
      ]
    }

Usage
-----
    # Trained model predictions
    python tools/transform_prediction_to_gtformat.py \
        --mode        trained \
        --gt_file     data/ground_truth.json \
        --pred_file   data/raw_prediction.json \
        --mapdict     data/mapdict_from_yolo_to_gt.json \
        --output      data/prediction/transformed_pred.json

    # Zero-shot predictions
    python tools/transform_prediction_to_gtformat.py \
        --mode        zero_shot \
        --gt_file     data/ground_truth.json \
        --pred_file   data/raw_prediction.json \
        --mapdict     data/mapdict_from_yolo_to_gt.json \
        --output      data/prediction/transformed_pred_zs.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Reaction transformation helpers
# ---------------------------------------------------------------------------

def _transform_reactions_trained(new_img: Dict,
                                  reactions: List[Dict],
                                  max_yolo_index: int) -> List[Dict]:
    """
    For trained-model predictions.
    Converts typed reaction items to integer bbox indices (0-based).
    Text/identifier items are appended as virtual bboxes with empty coords.
    """
    transformed = []
    next_idx = max_yolo_index

    for rxn in reactions:
        if not isinstance(rxn, dict):
            transformed.append({})
            continue

        new_rxn: Dict[str, List] = {}
        for role in ("reactants", "conditions", "products"):
            new_rxn[role] = []
            for item in rxn.get(role, []):
                if not isinstance(item, dict) or "type" not in item:
                    continue
                if item["type"] == "bbox":
                    raw_idx = item.get("index")
                    try:
                        idx = (raw_idx[0] if isinstance(raw_idx, list)
                               else int(raw_idx))
                        new_rxn[role].append(idx - 1)  # 1-based → 0-based
                    except (TypeError, ValueError):
                        pass
                elif item["type"] in ("txt", "idt"):
                    next_idx += 1
                    new_rxn[role].append(next_idx)
                    new_img["bboxes"].append({
                        "id": next_idx,
                        "category_id": 2,
                        "bbox": [0, 0, 0, 0],
                        "text": item.get("content", ""),
                    })
        transformed.append(new_rxn)

    return transformed


def _transform_reactions_zero_shot(new_img: Dict,
                                    reactions: List[Dict],
                                    max_yolo_index: int) -> List[Dict]:
    """
    For zero-shot predictions.
    Same logic as trained, but text items in the *conditions* role are
    kept as plain text entries; structure items fall back to sequence order
    when no index is provided.
    """
    transformed = []
    next_idx = max_yolo_index
    struct_counter = 0  # fallback sequential counter for structureless items

    for rxn in reactions:
        if not isinstance(rxn, dict):
            transformed.append({})
            continue

        new_rxn: Dict[str, List] = {}
        for role in ("reactants", "conditions", "products"):
            new_rxn[role] = []
            for item in rxn.get(role, []):
                if not isinstance(item, dict) or "type" not in item:
                    continue
                if item["type"] == "bbox":
                    raw_idx = item.get("index")
                    try:
                        idx = (raw_idx[0] if isinstance(raw_idx, list)
                               else int(raw_idx))
                        new_rxn[role].append(idx - 1)
                    except (TypeError, ValueError):
                        # No valid index → use sequential counter
                        new_rxn[role].append(struct_counter)
                        struct_counter += 1
                elif item["type"] in ("txt", "idt"):
                    next_idx += 1
                    new_rxn[role].append(next_idx)
                    new_img["bboxes"].append({
                        "id": next_idx,
                        "category_id": 2,
                        "bbox": [0, 0, 0, 0],
                        "text": item.get("content", ""),
                    })
        transformed.append(new_rxn)

    return transformed


def _filter_valid_reactions(image: Dict) -> Dict:
    """Clamp reaction indices to [0, len(bboxes) - 1]."""
    max_idx = len(image.get("bboxes", [])) - 1
    for rxn in image.get("reactions", []):
        for role in ("reactants", "conditions", "products"):
            rxn[role] = [i for i in rxn.get(role, []) if 0 <= i <= max_idx]
    return image


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def transform(gt_file: str, pred_file: str, mapdict_file: str,
              output_file: str, mode: str) -> None:
    # Load inputs
    with open(gt_file, encoding="utf-8") as f:
        base = json.load(f)
    with open(pred_file, encoding="utf-8") as f:
        yolo_pred = json.load(f)
    with open(mapdict_file, encoding="utf-8") as f:
        map_list = json.load(f)

    # Build file_name → {yolo_idx → info} lookup
    map_lookup: Dict[str, Dict] = {}
    for entry in map_list:
        if "file_name" not in entry:
            continue
        basename = entry["file_name"].split("/")[-1]
        idx_map = {}
        for yolo_id_str, details in entry.get("map_dict", {}).items():
            if isinstance(details, dict):
                idx_map[int(yolo_id_str)] = details
        map_lookup[basename] = idx_map

    out = {k: ([] if k == "images" else v) for k, v in base.items()}

    transform_fn = (_transform_reactions_trained
                    if mode == "trained"
                    else _transform_reactions_zero_shot)

    for item in yolo_pred:
        fname = item["file_name"].split("/")[-1]
        mapping = map_lookup.get(fname)
        pred_rxns = item.get("reactions") or []

        new_img: Dict = {"file_name": fname, "bboxes": [], "reactions": []}

        if mapping is None:
            # No mapping found: still need to transform reactions
            # Use transform function with empty max_yolo
            new_img["reactions"] = transform_fn(new_img, pred_rxns, -1)
        else:
            for yolo_idx, info in mapping.items():
                new_img["bboxes"].append({
                    "id": yolo_idx - 1,
                    "category_id": 1,
                    "bbox": info["yolo_bbox"],
                })
            max_yolo = max(mapping.keys(), default=0) - 1
            new_img["reactions"] = transform_fn(new_img, pred_rxns, max_yolo)

        out["images"].append(new_img)

    out["images"] = [_filter_valid_reactions(img) for img in out["images"]]

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(out['images'])} images → {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw model predictions to evaluation GT format."
    )
    parser.add_argument(
        "--mode", required=True, choices=["trained", "zero_shot"],
        help="'trained' for fine-tuned model; 'zero_shot' for baseline.",
    )
    parser.add_argument("--gt_file",   required=True,
                        help="Ground-truth JSON file (for schema / bbox IDs).")
    parser.add_argument("--pred_file", required=True,
                        help="Raw prediction JSON (from transform_jsonl_to_json.py).")
    parser.add_argument("--mapdict",   required=True,
                        help="Mapping dict JSON (from tools/generate_mapdict.py).")
    parser.add_argument("--output",    required=True,
                        help="Output path for the transformed prediction JSON.")
    args = parser.parse_args()

    transform(args.gt_file, args.pred_file, args.mapdict, args.output, args.mode)


if __name__ == "__main__":
    main()
