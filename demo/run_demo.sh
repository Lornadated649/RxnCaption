#!/bin/bash
# =============================================================================
# RxnCaption Full Demo (single machine, with eval + visualization)
# =============================================================================
# Runs the full pipeline on images in demo/sample_images/.
# Requires: GPU, MolYOLO weights, ms-swift installed.
#
# Usage:
#   cd RxnCaption
#   bash demo/run_demo.sh
#
# Optional: set environment variables before running:
#   MODEL=/path/to/local/RxnCaption-VL  bash demo/run_demo.sh
#   GT_FILE=/path/to/ground_truth.json  bash demo/run_demo.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_DIR="${SCRIPT_DIR}/sample_images"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
ANNOTATED_DIR="${OUTPUT_DIR}/annotated"
EVAL_RESULTS_DIR="${OUTPUT_DIR}/eval_results"
WEIGHTS="${REPO_ROOT}/molyolo/weights/MolYOLO.pt"
MODEL="${MODEL:-songjhPKU/RxnCaption-VL}"   # HuggingFace ID or local path
CONF=0.5
GT_FILE="${GT_FILE:-demo/sample_gt.json}"    # Ground truth for evaluation (optional)

# ── Check prerequisites ───────────────────────────────────────────────────────
NUM_IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')
if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "[ERROR] No images found in demo/sample_images/"
    echo "  Please copy 10-20 reaction diagram images there first."
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "[ERROR] MolYOLO weights not found at $WEIGHTS"
    echo "  Please copy MolYOLO.pt to molyolo/weights/"
    exit 1
fi

echo "Found $NUM_IMAGES images in demo/sample_images/"
echo ""

mkdir -p "$OUTPUT_DIR" "$EVAL_RESULTS_DIR"

# ── Step 1/7: MolYOLO Detection (GPU) ────────────────────────────────────────
echo "========================================"
echo " Step 1/7 — MolYOLO Detection"
echo "========================================"
python "${REPO_ROOT}/molyolo/predict.py" \
    --img_dir       "$IMAGE_DIR" \
    --weights       "$WEIGHTS" \
    --output_dir    "${OUTPUT_DIR}/molyolo" \
    --output_name   run \
    --conf          "$CONF" \
    --gpu_num       1 \
    --n_workers     1 \
    --visual_prompt

DET_JSON_DIR="${OUTPUT_DIR}/molyolo/run/json"

# ── Step 2/7: BIVP Annotation (CPU) ──────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 2/7 — BIVP Annotation"
echo "========================================"
python "${REPO_ROOT}/rxncaption/annotate.py" \
    --image_root_dir    "$IMAGE_DIR" \
    --det_json_root_dir "$DET_JSON_DIR" \
    --middle_root_dir   "$ANNOTATED_DIR" \
    --confidence_threshold "$CONF"

ANNOTATED_THRESH="${ANNOTATED_DIR}/threshold_${CONF}"

# ── Step 3/7: Build Inference JSONL (CPU) ─────────────────────────────────────
echo ""
echo "========================================"
echo " Step 3/7 — Build Inference JSONL"
echo "========================================"
EVAL_JSONL="${OUTPUT_DIR}/eval_input.jsonl"

export ANNOTATED_THRESH_DIR="$ANNOTATED_THRESH"
export EVAL_JSONL

python - << 'PYEOF'
import json, os
annotated_dir = os.environ["ANNOTATED_THRESH_DIR"]
output_jsonl  = os.environ["EVAL_JSONL"]

SYSTEM_PROMPT = (
    "You are a chemistry expert. Analyze the provided image which contains "
    "chemical reactions. Your task is to identify all chemical structures and "
    "relevant text (like reagents, conditions, identifiers). Then, organize "
    "them into a complete chemical reaction equation. Output the result as a "
    "JSON list, where each item represents a single reaction. Each reaction "
    "must contain 'reactants', 'conditions', and 'products'. Each of these is "
    "a list of objects. An object can be a structure represented as "
    '{"structure": <index>}, text as {"text": "<content>"}, or an '
    'identifier as {"identifier": "<content>"}. The <index> corresponds '
    "to the numeric label of a structure in the image. "
    "For example, if a reaction shows structure '1' and the text 'H2O' as "
    "reactants, 'heat' as a condition, and structure '2' as the product, "
    "your output for that single reaction would be: "
    '[{"reactants": [{"structure": 1}, {"text": "H2O"}], '
    '"conditions": [{"text": "heat"}], '
    '"products": [{"structure": 2}]}]. Output only the JSON.'
)
USER_PROMPT = "<image> Now output your JSON format result:"

count = 0
with open(output_jsonl, "w") as fout:
    for fname in sorted(os.listdir(annotated_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        entry = {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": USER_PROMPT},
                {"role": "assistant", "content": ""},
            ],
            "images": [os.path.join(annotated_dir, fname)],
        }
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        count += 1
print(f"Wrote {count} entries to {output_jsonl}")
PYEOF

# ── Step 4/7: VL Inference (GPU) ─────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 4/7 — RxnCaption-VL Inference"
echo "========================================"
INFER_OUTPUT="${OUTPUT_DIR}/infer_output.jsonl"

swift infer \
    --model           "$MODEL" \
    --model_type      qwen2_5_vl \
    --infer_backend   pt \
    --val_dataset     "$EVAL_JSONL" \
    --result_path     "$INFER_OUTPUT" \
    --max_batch_size  1 \
    --max_new_tokens  16384

# ── Step 5/7: JSONL to JSON (CPU) ────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 5/7 — Post-processing"
echo "========================================"
RAW_PRED="${OUTPUT_DIR}/raw_prediction.json"

python "${REPO_ROOT}/tools/transform_jsonl_to_json.py" \
    --input  "$INFER_OUTPUT" \
    --output "$RAW_PRED"

# ── Step 6/7: Generate Mapdict (CPU) ─────────────────────────────────────────
# This step and step 7 require ground truth. Skip if GT_FILE does not exist.
if [ ! -f "$GT_FILE" ]; then
    echo ""
    echo "========================================"
    echo " Steps 6-7 — Skipped (no ground truth)"
    echo "========================================"
    echo "  Ground truth file not found: $GT_FILE"
    echo "  To run evaluation, provide GT_FILE:"
    echo "    GT_FILE=/path/to/ground_truth.json bash demo/run_demo.sh"
    echo ""
    echo "========================================"
    echo " Demo Complete! (inference only)"
    echo "========================================"
    echo ""
    echo "Outputs:"
    echo "  Detection JSONs  : ${DET_JSON_DIR}/"
    echo "  VP images        : ${OUTPUT_DIR}/molyolo/run/vp_image/"
    echo "  Annotated images : ${ANNOTATED_THRESH}/"
    echo "  Raw predictions  : ${RAW_PRED}"
    exit 0
fi

echo ""
echo "========================================"
echo " Step 6/7 — Generate YOLO→GT Mapping"
echo "========================================"
MAPDICT="${OUTPUT_DIR}/mapdict_from_yolo_to_gt.json"

python "${REPO_ROOT}/tools/generate_mapdict.py" \
    --raw_gt_path   "$GT_FILE" \
    --yolo_path     "$DET_JSON_DIR" \
    --yolo_map_dict "$MAPDICT"

# ── Step 7/7: Evaluate + Visualize (CPU) ─────────────────────────────────────
echo ""
echo "========================================"
echo " Step 7/7 — Evaluate + Visualize"
echo "========================================"
TRANSFORMED_PRED="${EVAL_RESULTS_DIR}/transformed_prediction.json"

python "${REPO_ROOT}/tools/transform_prediction_to_gtformat.py" \
    --mode      trained \
    --gt_file   "$GT_FILE" \
    --pred_file "$RAW_PRED" \
    --mapdict   "$MAPDICT" \
    --output    "$TRANSFORMED_PRED"

python "${REPO_ROOT}/rxncaption/evaluate.py" \
    --ground_truth_file "$GT_FILE" \
    --pred_file         "$TRANSFORMED_PRED" \
    --image_base_path   "$IMAGE_DIR" \
    --output_dir        "$EVAL_RESULTS_DIR" \
    --mode              all \
    --limit_vis         20

echo ""
echo "========================================"
echo " Demo Complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Detection JSONs  : ${DET_JSON_DIR}/"
echo "  VP images        : ${OUTPUT_DIR}/molyolo/run/vp_image/"
echo "  Annotated images : ${ANNOTATED_THRESH}/"
echo "  Raw predictions  : ${RAW_PRED}"
echo "  Eval results     : ${EVAL_RESULTS_DIR}/"
echo "  Visualizations   : ${EVAL_RESULTS_DIR}/visualizations/"
