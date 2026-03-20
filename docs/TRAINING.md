# Training Guide

## Overview

RxnCaption-VL is fine-tuned from **Qwen2.5-VL-7B-Instruct** using [ms-swift](https://github.com/modelscope/ms-swift) supervised fine-tuning (SFT).

## Requirements

```bash
pip install ms-swift torch torchvision transformers accelerate deepspeed flash-attn
```

## Data Format

Training data is a JSONL file where each line is:

```json
{
  "messages": [
    {"role": "system",    "content": "<system_prompt>"},
    {"role": "user",      "content": "<image> Now output your JSON format result:"},
    {"role": "assistant", "content": "[{\"reactants\": [{\"structure\": 1}], ...}]"}
  ],
  "images": ["/absolute/path/to/annotated_image.png"]
}
```

See `tools/convert_to_qwen_format.py` for how to generate this from raw GT annotations.

## Prepare Training Data

```bash
bash scripts/prepare_data.sh \
    --raw_gt_json   data/ground_truth_ocr.json \
    --yolo_det_dir  data/det_json/ \
    --image_dir     data/annotated_images/ \
    --output_dir    data/processed/ \
    --dpi           400
```

## Launch Training (Single Node, 8 GPUs)

```bash
export NNODES=1
export NPROC_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

swift sft \
    --train_type full \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --dataset data/processed/train.jsonl \
    --val_dataset data/processed/val.jsonl \
    --system 'You are a helpful assistant.' \
    --max_length 16384 \
    --output_dir outputs/train/ \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --save_steps 30 \
    --eval_steps 30 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --freeze_vit false \
    --freeze_aligner false \
    --load_from_cache_file false
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen2.5-VL-7B-Instruct | Also tested with 3B |
| Train type | Full fine-tuning | All parameters updated |
| Max sequence length | 16384 | Long for multi-reaction images |
| Learning rate | 1e-5 | |
| Warmup ratio | 0.05 | |
| Batch size (effective) | 16 | 1 per device × 16 grad accumulation |
| Epochs | 10 | Early stop via eval F1 |
| Attn impl | flash_attn | Required for 16k context |
| DeepSpeed | Zero-2 | For 8-GPU training |

## Dataset Split

The **U-RxnDiagram-15k** dataset is split as:
- **Train**: ~99% (≈14,850 images, 4× augmented → ~59,000 JSONL lines)
- **Val**: ~1% (≈150 images)

The 4× augmentation applies to reversed/reversible reactions (see `tools/transform_yolo_detections.py`).

## Monitoring

Training logs are written to `outputs/train/train.log`.  
Use `tensorboard --logdir outputs/train/` to visualise loss curves.

## Upload to HuggingFace

```bash
# After training finishes:
huggingface-cli upload songjhPKU/RxnCaption-VL outputs/train/checkpoint-best/
```
