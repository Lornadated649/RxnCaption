"""
RxnCaption-VL Inference
=======================
Runs the fine-tuned Qwen2.5-VL vision-language model to predict chemical
reaction structures from annotated (visual-prompt) images.

The model is hosted on HuggingFace: songjhPKU/RxnCaption-VL

Inputs
------
- A JSONL evaluation dataset whose each line has the format produced by
  tools/convert_to_qwen_format.py:
      {
        "messages": [
          {"role": "system",  "content": "<system prompt>"},
          {"role": "user",    "content": "<image> Now output your JSON format result:"},
          {"role": "assistant","content": "<reference answer (ignored at inference)>"}
        ],
        "images": ["/absolute/path/to/annotated_image.png"]
      }

Outputs
-------
A JSONL file where every line is an inference result:
    {
      "images": [{"path": "..."}],
      "response": "<model output JSON string>"
    }

This JSONL file is then converted to the evaluation-ready JSON format by
tools/transform_jsonl_to_json.py.

Usage (single machine, one or more GPUs)
-----------------------------------------
    swift infer \
        --model  songjhPKU/RxnCaption-VL \
        --model_type qwen2_5_vl \
        --infer_backend pt \
        --val_dataset  /path/to/eval.jsonl \
        --result_path  /path/to/output.jsonl \
        --max_batch_size 1 \
        --max_new_tokens 16384

Or use the provided shell script:
    bash scripts/run_inference.sh

Notes
-----
- We rely on the `swift` CLI from the ms-swift library.
  Install with: pip install ms-swift
- For multi-GPU inference on a single node, set CUDA_VISIBLE_DEVICES
  before calling swift infer, e.g.:
      CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer ...
- The system prompt and user prompt templates are defined below and must
  match those used during fine-tuning.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (kept here as a reference; they live inside the JSONL data)
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# This module serves primarily as documentation.
# The actual inference is driven by the `swift infer` CLI call in
# scripts/run_inference.sh.
#
# For programmatic use, see the example below:
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE_USAGE = """
Example: programmatic inference with ms-swift

    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
    from swift.utils import get_model_tokenizer

    model_id = "songjhPKU/RxnCaption-VL"
    engine = PtEngine(model_id, model_type="qwen2_5_vl", max_batch_size=1)

    req = InferRequest(
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": USER_PROMPT},
        ],
        images=["path/to/annotated_image.png"],
    )

    cfg = RequestConfig(max_new_tokens=16384)
    response = engine.infer([req], cfg)[0].choices[0].message.content
    print(response)  # JSON string with predicted reactions
"""

if __name__ == "__main__":
    print(__doc__)
    print(EXAMPLE_USAGE)
