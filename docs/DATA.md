# Dataset Guide — U-RxnDiagram-15k

## Overview

**U-RxnDiagram-15k** (formerly RxnCaption-15k) is a dataset of ~15,000 chemical reaction diagram images extracted from scientific PDFs, each annotated with:

- Bounding boxes for all chemical entities (structures, text, identifiers, supplements)
- Complete reaction graphs linking reactants → conditions → products

HuggingFace: [songjhPKU/U-RxnDiagram-15k](https://huggingface.co/datasets/songjhPKU/U-RxnDiagram-15k)

## Diagram Types

| Type | Description |
|------|-------------|
| `single` | Single linear reaction (A → B) |
| `multiple` | Multiple parallel or sequential reactions |
| `tree` | Tree-structured multi-step synthesis |
| `graph` | Complex reaction network |

## JSON Schema

Both the ground-truth and prediction files use this schema:

```json
{
  "licenses": [],
  "info": {},
  "categories": [
    {"id": 1, "name": "structure"},
    {"id": 2, "name": "text"},
    {"id": 3, "name": "identifier"},
    {"id": 4, "name": "supplement"}
  ],
  "images": [
    {
      "id": 0,
      "file_name": "10.1002_foo_figure_0.png",
      "diagram_type": "single",
      "bboxes": [
        {"id": 0, "category_id": 1, "bbox": [x, y, w, h]},
        {"id": 1, "category_id": 2, "bbox": [x, y, w, h], "text": "NaOH"},
        {"id": 2, "category_id": 1, "bbox": [x, y, w, h]}
      ],
      "reactions": [
        {"reactants": [0], "conditions": [1], "products": [2]}
      ]
    }
  ]
}
```

### Category IDs

| category_id | Meaning |
|-------------|---------|
| 1 | Chemical structure (molecule, complex, etc.) |
| 2 | Text label (reagent, solvent, temperature, yield, ...) |
| 3 | Compound identifier (number, letter, code) |
| 4 | Supplementary annotation (ignored in evaluation) |

### Bbox format

`[x, y, width, height]` in pixels (COCO-style, 0-indexed top-left origin).

## Download

```python
from datasets import load_dataset
ds = load_dataset("songjhPKU/U-RxnDiagram-15k")
```

Or with HuggingFace CLI:

```bash
huggingface-cli download songjhPKU/U-RxnDiagram-15k --repo-type dataset --local-dir data/U-RxnDiagram-15k
```

## Evaluation Benchmarks

The paper evaluates on two held-out test sets:

| Benchmark | Source | # Images |
|-----------|--------|----------|
| **PDF-400** | Internal PDF corpus | 400 |
| **RxnScribe** | [RxnScribe dataset](https://github.com/Ozymandias314/openreactiondataset) | varies |

## Data Processing Pipeline

Raw PDF → cropped reaction images → YOLO detection → BIVP annotation → Qwen training JSONL

See `scripts/prepare_data.sh` for the full automated pipeline.
