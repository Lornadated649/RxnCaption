"""
Transform YOLO Detections into Augmented Training Data
=======================================================
Re-indexes bounding-box IDs in the source GT JSON from GT-space to
YOLO reading-order space (using the mapping produced by generate_mapdict.py),
then optionally duplicates samples that contain:

  * Horizontal reversed reactions  (reactants right of products)
  * Vertical reversed reactions    (reactants below products)
  * Reversible reactions           (A→B and B→A both present)

These augmentation strategies improve model robustness.

Usage
-----
    python tools/transform_yolo_detections.py \
        --source_json   data/ground_truth_ocr.json \
        --mapdict_json  data/mapdict_from_yolo_to_gt.json \
        --output_json   data/gt_enhanced.json \
        --discard_log   data/discard_log.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration constants (can also be passed as CLI flags)
# ---------------------------------------------------------------------------

IOU_THRESHOLD = 0.50
KEEP_UNMATCHED_ITEMS = False        # drop images whose GT→YOLO mapping fails

DUPLICATE_HORIZONTAL_REVERSED = True
HORIZONTAL_REVERSED_FACTOR = 2

DUPLICATE_VERTICAL_REVERSED = True
VERTICAL_REVERSED_FACTOR = 2

DUPLICATE_REVERSIBLE_REACTIONS = True
REVERSIBLE_REACTION_FACTOR = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DiscardLog:
    """Accumulates per-image discard reasons and writes them to a JSON file."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._entries: List[Dict] = []

    def record(self, image_id, file_name, reaction_idx,
               role, obj_id, reason, iou_detail=None) -> None:
        entry = {
            "image_id": image_id, "file_name": file_name,
            "reaction_index": reaction_idx, "role": role,
            "obj_id": obj_id, "reason": reason,
        }
        if iou_detail is not None:
            entry["iou_detail"] = iou_detail
        self._entries.append(entry)

    def save(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, ensure_ascii=False)
        print(f"Discard log saved: {self.log_path} ({len(self._entries)} entries)")


def _strip_prefix(file_name: str) -> str:
    """Remove 'figure/' or 'table/' directory prefixes from a filename."""
    file_name = file_name.replace(".jpg", ".png")
    for prefix in ("figure/", "table/"):
        if file_name.startswith(prefix):
            return file_name[len(prefix):]
    return file_name


def _is_reversible(reactions: List[Dict]) -> bool:
    """Return True if the reaction list contains a reversible pair (A→B and B→A)."""
    for i in range(len(reactions)):
        for j in range(i + 1, len(reactions)):
            r1, r2 = reactions[i], reactions[j]
            if (set(r1.get("reactants", [])) == set(r2.get("products", [])) and
                    set(r1.get("products", [])) == set(r2.get("reactants", []))):
                return True
    return False


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def process_item(item: Dict, discard_log: DiscardLog,
                 ground_truth_map: Dict) -> Tuple[Optional[Dict], int, Tuple]:
    """
    Re-map bbox IDs in *item* from GT-space to YOLO reading-order space.

    Returns
    -------
    (transformed_item | None, duplication_factor, (h_rev, v_rev, is_rev))
    Returns (None, 1, (F,F,F)) when the item must be discarded.
    """
    file_name = item.get("file_name", "")
    image_id  = item.get("id")

    stripped = _strip_prefix(file_name)
    image_gt = ground_truth_map.get(os.path.basename(stripped))
    if not image_gt:
        logging.warning("GT not found for %s, skipping.", stripped)
        return None, 1, (False, False, False)

    map_dict: Dict = image_gt.get("map_dict", {})
    if not map_dict:
        discard_log.record(image_id, file_name, -1, "N/A", -1,
                           "No map_dict in ground truth.")
        return None, 1, (False, False, False)

    # Build old_id → new_id mapping from the map_dict
    old_to_new: Dict[int, int] = {}
    new_id_set: set = set()
    for yolo_id_str, info in map_dict.items():
        if ("gt_id" in info and "yolo_bbox" in info and info["yolo_bbox"]):
            try:
                new_id = int(yolo_id_str)
                old_to_new[info["gt_id"]] = new_id
                new_id_set.add(new_id)
            except (ValueError, TypeError):
                discard_log.record(image_id, file_name, -1, "N/A", -1,
                                   f"Invalid yolo_id '{yolo_id_str}'.")
                return None, 1, (False, False, False)

    # Assign fallback IDs for unmapped old IDs
    all_old_ids = {b["id"] for b in item.get("bboxes", [])}
    unmapped = all_old_ids - set(old_to_new.keys())
    next_id = (max(new_id_set) + 1) if new_id_set else 0
    for old_id in sorted(unmapped):
        while next_id in new_id_set:
            next_id += 1
        old_to_new[old_id] = next_id
        new_id_set.add(next_id)
        next_id += 1

    # Re-map bboxes
    new_bboxes = []
    for bbox in item.get("bboxes", []):
        old_id = bbox["id"]
        if old_id not in old_to_new:
            discard_log.record(image_id, file_name, -1, "N/A", old_id,
                               "Bbox ID not found in old_to_new.")
            return None, 1, (False, False, False)
        nb = dict(bbox)
        nb["id"] = old_to_new[old_id]
        new_bboxes.append(nb)

    # Re-map reactions
    new_reactions = []
    for i, rxn in enumerate(item.get("reactions", [])):
        new_rxn: Dict[str, List] = {}
        valid = True
        for role in ("reactants", "conditions", "products"):
            new_ids = []
            for old_id in rxn.get(role, []):
                if old_id not in old_to_new:
                    discard_log.record(image_id, file_name, i, role, old_id,
                                       "Reaction ID not in mapping.")
                    valid = False
                    break
                new_ids.append(old_to_new[old_id])
            if not valid:
                break
            new_rxn[role] = new_ids
        if not valid:
            return None, 1, (False, False, False)
        new_reactions.append(new_rxn)

    result = dict(item)
    result["bboxes"] = new_bboxes
    result["reactions"] = new_reactions

    # Detect reversed / reversible reactions for augmentation
    h_rev = v_rev = is_rev = False
    if DUPLICATE_REVERSIBLE_REACTIONS and _is_reversible(item["reactions"]):
        is_rev = True

    if (DUPLICATE_HORIZONTAL_REVERSED or DUPLICATE_VERTICAL_REVERSED) and new_reactions and new_bboxes:
        id2bbox = {b["id"]: b["bbox"] for b in new_bboxes}
        for rxn in new_reactions:
            r_boxes = [id2bbox[rid] for rid in rxn.get("reactants", []) if rid in id2bbox]
            p_boxes = [id2bbox[pid] for pid in rxn.get("products",  []) if pid in id2bbox]
            if not r_boxes or not p_boxes:
                continue
            # Horizontal: reactants all to the right of products
            if DUPLICATE_HORIZONTAL_REVERSED and not h_rev:
                if min(b[0] for b in r_boxes) > max(b[0] + b[2] for b in p_boxes):
                    h_rev = True
            # Vertical: reactants all below products
            if DUPLICATE_VERTICAL_REVERSED and not v_rev:
                if min(b[1] for b in r_boxes) > max(b[1] + b[3] for b in p_boxes):
                    v_rev = True
            if h_rev and v_rev:
                break

    factor = 1
    if h_rev:
        factor = max(factor, HORIZONTAL_REVERSED_FACTOR)
    if v_rev:
        factor = max(factor, VERTICAL_REVERSED_FACTOR)
    if is_rev:
        factor = max(factor, REVERSIBLE_REACTION_FACTOR)

    return result, factor, (h_rev, v_rev, is_rev)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform raw GT JSON into YOLO-indexed training data."
    )
    parser.add_argument("--source_json",  required=True,
                        help="Input ground-truth JSON with original bbox IDs.")
    parser.add_argument("--mapdict_json", required=True,
                        help="Mapping JSON produced by tools/generate_mapdict.py.")
    parser.add_argument("--output_json",  required=True,
                        help="Path for the transformed output JSON.")
    parser.add_argument("--discard_log",  default="discard_log.json",
                        help="Path for the discard-log JSON.")
    args = parser.parse_args()

    print("Loading mapping dictionary …")
    with open(args.mapdict_json, encoding="utf-8") as f:
        mapdict_list = json.load(f)
    gt_map = {os.path.basename(item["file_name"]): item for item in mapdict_list}
    print(f"  {len(gt_map)} entries loaded.")

    print(f"Loading source data: {args.source_json}")
    with open(args.source_json, encoding="utf-8") as f:
        source = json.load(f)

    discard_log = DiscardLog(args.discard_log)
    transformed: List[Dict] = []
    h_count = v_count = rev_count = 0

    for item in tqdm(source.get("images", []), desc="Processing"):
        if not item.get("reactions"):
            if item.get("bboxes"):
                transformed.append(item)
            continue

        result, factor, (h_rev, v_rev, is_rev) = process_item(
            item, discard_log, gt_map
        )
        if result:
            if not result.get("bboxes") and not result.get("reactions"):
                continue
            if h_rev: h_count += 1
            if v_rev: v_count += 1
            if is_rev: rev_count += 1
            for _ in range(factor):
                transformed.append(result)

    output = {
        "licenses":   source.get("licenses", []),
        "info":       source.get("info", {}),
        "categories": source.get("categories", []),
        "images":     transformed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    discard_log.save()
    print(f"\nOutput written: {args.output_json}")
    print(f"  Total transformed images : {len(transformed)}")
    print(f"  Horizontally reversed    : {h_count}")
    print(f"  Vertically reversed      : {v_count}")
    print(f"  Reversible reactions     : {rev_count}")


if __name__ == "__main__":
    main()
