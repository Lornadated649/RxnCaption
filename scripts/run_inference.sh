#!/bin/bash
# =============================================================================
# RxnCaption Full Inference Pipeline
# =============================================================================
# Runs MolYOLO detection → BIVP annotation → VL model inference → format
# conversion, all on a single machine (no SLURM).
#
# Requirements:
#   - PyTorch + CUDA
#   - ultralytics (YOLOv10 fork in molyolo/ultralytics/)
#   - ms-swift  (pip install ms-swift)
#   - Pillow, tqdm, opencv-python, openpyxl
#
# Usage:
#   bash scripts/run_inference.sh \
#       --image_dir  /path/to/raw_images \
#       --output_dir /path/to/outputs \
#       [--gpu_num 4] [--conf 0.5]
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_DIR=""
OUTPUT_DIR="${REPO_ROOT}/outputs"
GPU_NUM=1
CONF=0.5
DPI=400
MODEL_PATH="songjhPKU/RxnCaption-VL"   # HuggingFace model ID or local path
WEIGHTS="${REPO_ROOT}/molyolo/weights/MolYOLO.pt"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --image_dir <dir> [--output_dir <dir>] [--gpu_num N] [--conf F] [--dpi N] [--model <hf_id_or_path>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_dir)   IMAGE_DIR="$2";   shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --gpu_num)     GPU_NUM="$2";     shift 2 ;;
        --conf)        CONF="$2";        shift 2 ;;
        --dpi)         DPI="$2";         shift 2 ;;
        --model)       MODEL_PATH="$2";  shift 2 ;;
        --weights)     WEIGHTS="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$IMAGE_DIR" ]] && { echo "[ERROR] --image_dir is required."; usage; }

mkdir -p "$OUTPUT_DIR"

# ── Step 1: MolYOLO Detection ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 1/4 — MolYOLO Detection"
echo "========================================"

DET_JSON_DIR="${OUTPUT_DIR}/det_json"
python "${REPO_ROOT}/molyolo/predict.py" \
    --img_dir       "$IMAGE_DIR" \
    --weights       "$WEIGHTS" \
    --output_dir    "${OUTPUT_DIR}/molyolo" \
    --output_name   "run" \
    --conf          "$CONF" \
    --gpu_num       "$GPU_NUM" \
    --n_workers     "$GPU_NUM" \
    --visual_prompt

# molyolo/predict.py writes JSONs to ${OUTPUT_DIR}/molyolo/run/json/
DET_JSON_DIR="${OUTPUT_DIR}/molyolo/run/json"

# ── Step 2: BIVP Annotation ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 2/4 — BIVP Annotation"
echo "========================================"

ANNOTATED_DIR="${OUTPUT_DIR}/annotated"
python "${REPO_ROOT}/rxncaption/annotate.py" \
    --image_root_dir    "$IMAGE_DIR" \
    --det_json_root_dir "$DET_JSON_DIR" \
    --middle_root_dir   "$ANNOTATED_DIR" \
    --confidence_threshold "$CONF"

ANNOTATED_THRESH_DIR="${ANNOTATED_DIR}/threshold_${CONF}"

# ── Step 3: Build eval JSONL (no GT labels needed — pure inference) ───────────
echo ""
echo "========================================"
echo " Step 3/4 — Building inference JSONL"
echo "========================================"

export ANNOTATED_THRESH_DIR
export EVAL_JSONL="${OUTPUT_DIR}/eval_input.jsonl"

python - <<'PYEOF'
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
    "{\"structure\": <index>}, text as {\"text\": \"<content>\"}, or an "
    "identifier as {\"identifier\": \"<content>\"}. The <index> corresponds "
    "to the numeric label of a structure in the image. "
    "For example, if a reaction shows structure '1' and the text 'H2O' as "
    "reactants, 'heat' as a condition, and structure '2' as the product, "
    "your output for that single reaction would be: "
    "[{\"reactants\": [{\"structure\": 1}, {\"text\": \"H2O\"}], "
    "\"conditions\": [{\"text\": \"heat\"}], "
    "\"products\": [{\"structure\": 2}]}]. Output only the JSON."
)
USER_PROMPT = "<image> Now output your JSON format result:"

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

print(f"Wrote eval JSONL: {output_jsonl}")
PYEOF

# ── Step 4: VL Model Inference ────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Step 4/4 — RxnCaption-VL Inference"
echo "========================================"

INFER_OUTPUT_JSONL="${OUTPUT_DIR}/infer_output.jsonl"

swift infer \
    --model           "$MODEL_PATH" \
    --model_type      qwen2_5_vl \
    --infer_backend   pt \
    --val_dataset     "$EVAL_JSONL" \
    --result_path     "$INFER_OUTPUT_JSONL" \
    --max_batch_size  1 \
    --max_new_tokens  16384

echo ""
echo "========================================"
echo " Post-processing"
echo "========================================"

RAW_PRED_JSON="${OUTPUT_DIR}/raw_prediction.json"
python "${REPO_ROOT}/tools/transform_jsonl_to_json.py" \
    --input  "$INFER_OUTPUT_JSONL" \
    --output "$RAW_PRED_JSON"

echo ""
echo "All done!"
echo "  Raw predictions (typed format) : $RAW_PRED_JSON"
echo "  To evaluate, run scripts/run_eval.sh"
