"""
Convert Remap JSON → Qwen Training JSONL
=========================================
Converts a re-mapped GT JSON file (output of transform_yolo_detections.py)
into the JSONL format expected by ms-swift for supervised fine-tuning of
Qwen2.5-VL.

Each output line:
    {
      "messages": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "<image> Now output your JSON format result:"},
        {"role": "assistant", "content": "<JSON reaction list>"}
      ],
      "images": ["/path/to/annotated_image.png"]
    }

The assistant content uses the following schema per reaction::

    [
      {
        "reactants":  [{"structure": 1}, {"text": "H2O"}],
        "conditions": [{"text": "heat"}],
        "products":   [{"structure": 2}]
      }
    ]

  ``category_id`` mapping: 1=structure, 2=text, 3=identifier, 4=supplement.
  Category 4 (supplement) items are silently dropped.

Usage
-----
    python tools/convert_to_qwen_format.py \
        --input_json   data/gt_enhanced.json \
        --output_jsonl data/train.jsonl \
        --image_base_path /path/to/annotated_images
"""

import argparse
import json
import os

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a chemistry expert. "
    "Analyze the provided image which contains chemical reactions. "
    "Your task is to identify all chemical structures and relevant text "
    "(like reagents, conditions, identifiers). "
    "Then, organize them into a complete chemical reaction equation. "
    "Output the result as a JSON list, where each item represents a single reaction. "
    "Each reaction must contain 'reactants', 'conditions', and 'products'. "
    "Each of these is a list of objects. "
    "An object can be a structure represented as {\"structure\": <index>}, "
    "text as {\"text\": \"<content>\"}, "
    "or an identifier as {\"identifier\": \"<content>\"}. "
    "The <index> corresponds to the numeric label of a structure in the image. "
    "For example, if a reaction shows structure '1' and the text 'H2O' as reactants, "
    "'heat' as a condition, and structure '2' as the product, "
    "your output for that single reaction would be: "
    "`[{\"reactants\": [{\"structure\": 1}, {\"text\": \"H2O\"}], "
    "\"conditions\": [{\"text\": \"heat\"}], "
    "\"products\": [{\"structure\": 2}]}]`. "
    "Output only the JSON."
)

USER_PROMPT = "<image> Now output your JSON format result:"


# ---------------------------------------------------------------------------
# Reaction serialisation
# ---------------------------------------------------------------------------

def _bbox_to_part(bbox: dict) -> dict | None:
    """Convert a single bbox record to its Qwen reaction-part representation."""
    cat = bbox.get("category_id")
    text = bbox.get("text", "")
    if cat == 1:
        return {"structure": bbox["id"]}
    if cat == 2:
        return {"text": text}
    if cat == 3:
        return {"identifier": text}
    # category_id 4 (supplement) and unknown – skip
    return None


def _build_qwen_entry(image_data: dict, image_base_path: str) -> dict | None:
    """
    Build a single JSONL entry for one image.
    Returns None if the image has no reactions.
    """
    bbox_map = {b["id"]: b for b in image_data.get("bboxes", [])}
    reactions = image_data.get("reactions", [])

    qwen_reactions = []
    for rxn in reactions:
        new_rxn: dict = {"reactants": [], "conditions": [], "products": []}
        for role in ("reactants", "conditions", "products"):
            for bid in rxn.get(role, []):
                if bid not in bbox_map:
                    continue
                part = _bbox_to_part(bbox_map[bid])
                if part is not None:
                    new_rxn[role].append(part)
        qwen_reactions.append(new_rxn)

    # Build absolute image path (swap .jpg → .png)
    fname = image_data.get("file_name", "").replace(".jpg", ".png")
    image_path = os.path.join(image_base_path, fname)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": json.dumps(qwen_reactions, ensure_ascii=False)},
        ],
        "images": [image_path],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert re-mapped GT JSON to Qwen2.5-VL training JSONL."
    )
    parser.add_argument("--input_json",     required=True,
                        help="Input JSON (output of transform_yolo_detections.py).")
    parser.add_argument("--output_jsonl",   required=True,
                        help="Path for the output JSONL file.")
    parser.add_argument("--image_base_path", required=True,
                        help="Directory containing the annotated (visual-prompt) images.")
    args = parser.parse_args()

    print(f"Loading {args.input_json} …")
    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for image_data in tqdm(images, desc="Converting"):
            entry = _build_qwen_entry(image_data, args.image_base_path)
            if entry is not None:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

    print(f"\nDone. {written} entries written to {args.output_jsonl}")


if __name__ == "__main__":
    main()
