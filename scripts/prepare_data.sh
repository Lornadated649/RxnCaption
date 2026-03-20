#!/bin/bash
# =============================================================================
# Data Preparation Pipeline
# =============================================================================
# Produces the training JSONL from raw annotations:
#
#   1. Generate YOLO→GT mapping dict (generate_mapdict.py)
#   2. Re-index GT bboxes into YOLO reading-order (transform_yolo_detections.py)
#   3. Convert to Qwen training JSONL  (convert_to_qwen_format.py)
#
# Usage:
#   bash scripts/prepare_data.sh \
#       --raw_gt_json     data/ground_truth_ocr.json \
#       --yolo_det_dir    data/det_json/ \
#       --image_dir       data/annotated_images/ \
#       --output_dir      data/processed/ \
#       [--dpi 400]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
RAW_GT_JSON=""
YOLO_DET_DIR=""
IMAGE_DIR=""
OUTPUT_DIR="${REPO_ROOT}/data/processed"
DPI=400

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --raw_gt_json <f> --yolo_det_dir <d> --image_dir <d> [options]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw_gt_json)  RAW_GT_JSON="$2";  shift 2 ;;
        --yolo_det_dir) YOLO_DET_DIR="$2"; shift 2 ;;
        --image_dir)    IMAGE_DIR="$2";    shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for var in RAW_GT_JSON YOLO_DET_DIR IMAGE_DIR; do
    [[ -z "${!var}" ]] && { echo "[ERROR] --${var,,} is required."; usage; }
done

mkdir -p "$OUTPUT_DIR"

# ── Step 1: Generate mapping dict ─────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 1/3 — Generate YOLO→GT Mapping"
echo "========================================"

MAPDICT="${OUTPUT_DIR}/mapdict_from_yolo_to_gt.json"
python "${REPO_ROOT}/tools/generate_mapdict.py" \
    --raw_gt_path   "$RAW_GT_JSON" \
    --yolo_path     "$YOLO_DET_DIR" \
    --yolo_map_dict "$MAPDICT" \
    --dpi           "$DPI"

# ── Step 2: Re-index into YOLO space ─────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 2/3 — Transform YOLO Detections"
echo "========================================"

GT_ENHANCED="${OUTPUT_DIR}/gt_enhanced.json"
DISCARD_LOG="${OUTPUT_DIR}/discard_log.json"
python "${REPO_ROOT}/tools/transform_yolo_detections.py" \
    --source_json   "$RAW_GT_JSON" \
    --mapdict_json  "$MAPDICT" \
    --output_json   "$GT_ENHANCED" \
    --discard_log   "$DISCARD_LOG"

# ── Step 3: Convert to Qwen JSONL ────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 3/3 — Convert to Qwen JSONL"
echo "========================================"

TRAIN_JSONL="${OUTPUT_DIR}/train.jsonl"
python "${REPO_ROOT}/tools/convert_to_qwen_format.py" \
    --input_json      "$GT_ENHANCED" \
    --output_jsonl    "$TRAIN_JSONL" \
    --image_base_path "$IMAGE_DIR"

echo ""
echo "Data preparation complete!"
echo "  Mapping dict  : $MAPDICT"
echo "  Enhanced GT   : $GT_ENHANCED"
echo "  Discard log   : $DISCARD_LOG"
echo "  Training JSONL: $TRAIN_JSONL"
