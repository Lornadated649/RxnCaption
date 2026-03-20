# Mini Demo

This demo lets you verify the full RxnCaption pipeline on a small set of images.

## Setup

1. **Copy test images**: Place 10-20 reaction diagram images (PNG/JPG) into `demo/sample_images/`.
   You can use images from the U-RxnDiagram-15k test set or your own.

2. **Prepare ground truth** (optional, for evaluation):
   - If using U-RxnDiagram-15k test images, extract the corresponding entries from the dataset's `ground_truth.json` into `demo/sample_gt.json`.
   - The format should match the COCO-like schema (see `docs/DATA.md`).
   - If no ground truth is provided, the demo will run inference only (steps 1-5) and skip evaluation.

3. **Ensure weights are available**:
   - `molyolo/weights/MolYOLO.pt` — MolYOLO checkpoint
   - `songjhPKU/RxnCaption-VL` — auto-downloaded from HuggingFace (or set a local path)

## Run the Demo

### Single machine (requires GPU)

```bash
# Basic usage (inference only, no evaluation)
bash demo/run_demo.sh

# With evaluation (provide ground truth)
GT_FILE=demo/sample_gt.json bash demo/run_demo.sh

# With a local model checkpoint
MODEL=/path/to/RxnCaption-VL bash demo/run_demo.sh
```

### SLURM cluster

```bash
# Edit demo/run_demo_slurm.sh to set your PARTITION and GT_FILE, then:
bash demo/run_demo_slurm.sh
```

### What the demo runs (7 steps)

1. **MolYOLO detection** (GPU) → `demo/outputs/molyolo/`
2. **BIVP annotation** (CPU) → `demo/outputs/annotated/`
3. **Build inference JSONL** (CPU) → `demo/outputs/eval_input.jsonl`
4. **RxnCaption-VL inference** (GPU) → `demo/outputs/infer_output.jsonl`
5. **Post-processing** (CPU) → `demo/outputs/raw_prediction.json`
6. **Generate YOLO→GT mapping** (CPU, requires GT) → `demo/outputs/mapdict_from_yolo_to_gt.json`
7. **Evaluation + visualization** (CPU, requires GT) → `demo/outputs/eval_results/`

### Step-by-step (if you want to run parts separately)

```bash
# Step 1: Detection only (GPU needed)
python molyolo/predict.py \
    --img_dir demo/sample_images \
    --weights molyolo/weights/MolYOLO.pt \
    --output_dir demo/outputs/molyolo \
    --output_name run --conf 0.5 --gpu_num 1 --visual_prompt

# Step 2: BIVP annotation (CPU only)
python rxncaption/annotate.py \
    --image_root_dir demo/sample_images \
    --det_json_root_dir demo/outputs/molyolo/run/json \
    --middle_root_dir demo/outputs/annotated \
    --confidence_threshold 0.5

# Step 3-4: Build JSONL + VL inference (GPU needed)
# See demo/run_demo.sh for details

# Step 5: Post-processing
python tools/transform_jsonl_to_json.py \
    --input demo/outputs/infer_output.jsonl \
    --output demo/outputs/raw_prediction.json

# Step 6: Generate mapping (requires GT)
python tools/generate_mapdict.py \
    --raw_gt_path demo/sample_gt.json \
    --yolo_path demo/outputs/molyolo/run/json \
    --yolo_map_dict demo/outputs/mapdict_from_yolo_to_gt.json

# Step 7: Evaluation with visualization (requires GT)
python tools/transform_prediction_to_gtformat.py \
    --mode trained \
    --gt_file demo/sample_gt.json \
    --pred_file demo/outputs/raw_prediction.json \
    --mapdict demo/outputs/mapdict_from_yolo_to_gt.json \
    --output demo/outputs/eval_results/transformed_prediction.json

python rxncaption/evaluate.py \
    --ground_truth_file demo/sample_gt.json \
    --pred_file demo/outputs/eval_results/transformed_prediction.json \
    --image_base_path demo/sample_images \
    --output_dir demo/outputs/eval_results \
    --mode all --limit_vis 20
```

## Expected Output

After running the full pipeline, you should see:
- `demo/outputs/molyolo/run/json/` — per-image detection JSONs
- `demo/outputs/molyolo/run/vp_image/` — visual prompt images from YOLO
- `demo/outputs/annotated/` — BIVP-annotated images
- `demo/outputs/infer_output.jsonl` — raw model output
- `demo/outputs/raw_prediction.json` — parsed predictions

If ground truth was provided:
- `demo/outputs/eval_results/` — Excel report + visualization PNGs + Markdown report
- `demo/outputs/eval_results/visualizations/` — per-image error analysis
