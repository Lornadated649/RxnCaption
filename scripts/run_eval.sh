#!/bin/bash
# =============================================================================
# RxnCaption Evaluation Pipeline
# =============================================================================
# Transforms raw predictions into evaluation format, then computes
# Hard / Soft / Hybrid precision, recall, and F1.
#
# Usage:
#   bash scripts/run_eval.sh \
#       --gt_file        data/ground_truth.json \
#       --raw_pred_file  outputs/raw_prediction.json \
#       --mapdict        data/mapdict_from_yolo_to_gt.json \
#       --image_dir      data/images \
#       --output_dir     results/ \
#       [--mode          all]             # overall | export_excel | export_f1_rank | visualize | all
#       [--pred_mode     trained]         # trained | zero_shot
#       [--limit_vis     10]              # max visualization cases (<=0 = no limit)
#       [--vis_range     1 5]             # only visualize cases with MIN <= FP+FN <= MAX
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
GT_FILE=""
RAW_PRED_FILE=""
MAPDICT=""
IMAGE_DIR=""
OUTPUT_DIR="${REPO_ROOT}/results"
EVAL_MODE="all"
PRED_MODE="trained"
LIMIT_VIS=-1
VIS_RANGE_MIN=""
VIS_RANGE_MAX=""

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --gt_file <f> --raw_pred_file <f> --mapdict <f> --image_dir <d> [options]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt_file)       GT_FILE="$2";       shift 2 ;;
        --raw_pred_file) RAW_PRED_FILE="$2"; shift 2 ;;
        --mapdict)       MAPDICT="$2";       shift 2 ;;
        --image_dir)     IMAGE_DIR="$2";     shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --mode)          EVAL_MODE="$2";     shift 2 ;;
        --pred_mode)     PRED_MODE="$2";     shift 2 ;;
        --limit_vis)     LIMIT_VIS="$2";     shift 2 ;;
        --vis_range)     VIS_RANGE_MIN="$2"; VIS_RANGE_MAX="$3"; shift 3 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for var in GT_FILE RAW_PRED_FILE MAPDICT IMAGE_DIR; do
    [[ -z "${!var}" ]] && { echo "[ERROR] --${var,,} is required."; usage; }
done

mkdir -p "$OUTPUT_DIR"

# ── Step 1: Transform predictions to GT format ────────────────────────────────
echo ""
echo "========================================"
echo " Step 1/2 — Transform Predictions"
echo "========================================"

TRANSFORMED_PRED="${OUTPUT_DIR}/transformed_prediction.json"
python "${REPO_ROOT}/tools/transform_prediction_to_gtformat.py" \
    --mode      "$PRED_MODE" \
    --gt_file   "$GT_FILE" \
    --pred_file "$RAW_PRED_FILE" \
    --mapdict   "$MAPDICT" \
    --output    "$TRANSFORMED_PRED"

# ── Step 2: Evaluation ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 2/2 — Evaluate"
echo "========================================"

VIS_ARGS=""
if [[ "$LIMIT_VIS" != "-1" ]]; then
    VIS_ARGS="$VIS_ARGS --limit_vis $LIMIT_VIS"
fi
if [[ -n "$VIS_RANGE_MIN" ]] && [[ -n "$VIS_RANGE_MAX" ]]; then
    VIS_ARGS="$VIS_ARGS --vis_range $VIS_RANGE_MIN $VIS_RANGE_MAX"
fi

python "${REPO_ROOT}/rxncaption/evaluate.py" \
    --ground_truth_file "$GT_FILE" \
    --pred_file         "$TRANSFORMED_PRED" \
    --image_base_path   "$IMAGE_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --mode              "$EVAL_MODE" \
    $VIS_ARGS

echo ""
echo "Evaluation complete. Results in: $OUTPUT_DIR"
