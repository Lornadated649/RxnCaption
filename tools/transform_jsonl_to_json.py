"""
Convert Inference JSONL → Evaluation-ready JSON
================================================
Parses the raw JSONL output produced by ``swift infer`` and converts it
to the unified JSON format expected by rxncaption/evaluate.py and
tools/transform_prediction_to_gtformat.py.

Input JSONL (one line per image)::

    {
      "images": [{"path": "/path/to/annotated/foo.png"}],
      "response": "[{\"reactants\": [{\"structure\": 1}], ...}]"
    }

Output JSON::

    [
      {"file_name": "subdir/foo.png", "reactions": [...]},
      ...
    ]

Each reaction item is converted to the internal typed format::

    {"reactants": [{"type": "bbox", "index": 1}],
     "conditions": [{"type": "txt",  "content": "heat"}],
     "products":   [{"type": "bbox", "index": 2}]}

Usage
-----
    python tools/transform_jsonl_to_json.py \
        --input  /path/to/infer_output.jsonl \
        --output /path/to/raw_prediction.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Allow very large integer strings (Python 3.11+)
try:
    sys.set_int_max_str_digits(100_000)
except AttributeError:
    pass  # Python < 3.11, no limit enforced


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_ints(obj: Any) -> Any:
    """Recursively convert short numeric strings back to int."""
    if isinstance(obj, dict):
        return {k: _coerce_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_ints(v) for v in obj]
    if isinstance(obj, str) and len(obj) < 100:
        try:
            return int(obj)
        except (ValueError, OverflowError):
            pass
    return obj


def _parse_response(response_str: str) -> List[Dict]:
    """
    Robustly parse the model's JSON response string.
    Falls back to object-by-object manual extraction on malformed JSON.
    """
    raw: List[Dict] = []

    # Fast path: valid JSON
    try:
        raw = json.loads(response_str.replace('\\"', '\\\\"'))
    except (json.JSONDecodeError, ValueError) as exc:
        # Handle large-integer edge cases
        try:
            raw = json.loads(response_str.replace('\\"', '\\\\"'),
                             parse_int=str)
            raw = _coerce_ints(raw)
        except Exception:
            pass

    # Slow path: extract objects one by one
    if not raw:
        pos = 0
        while True:
            start = response_str.find("{", pos)
            if start == -1:
                break
            depth, end = 1, -1
            for i in range(start + 1, len(response_str)):
                if response_str[i] == "{":
                    depth += 1
                elif response_str[i] == "}":
                    depth -= 1
                if depth == 0:
                    end = i
                    break
            if end == -1:
                break
            try:
                obj = json.loads(response_str[start: end + 1], parse_int=str)
                raw.append(_coerce_ints(obj))
            except (json.JSONDecodeError, ValueError):
                pass
            pos = end + 1

    return raw


def _convert_item(item: Dict) -> Optional[Dict]:
    """Convert one raw reaction-part dict to the typed internal format."""
    if "structure" in item:
        idx = item["structure"]
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except (ValueError, OverflowError):
                pass
        return {"type": "bbox", "index": idx}
    if "text" in item:
        return {"type": "txt", "content": item["text"]}
    if "identifier" in item:
        return {"type": "txt", "content": item["identifier"]}
    return None


def _transform_reactions(raw_reactions: List[Dict]) -> List[Dict]:
    """Convert a list of raw reaction dicts into typed internal format."""
    result = []
    for rxn in raw_reactions:
        if not isinstance(rxn, dict):
            continue
        new_rxn: Dict[str, List] = {}
        for role in ("reactants", "conditions", "products"):
            if role in rxn:
                converted = [_convert_item(it) for it in rxn.get(role, [])
                             if isinstance(it, dict)]
                new_rxn[role] = [c for c in converted if c is not None]
        if new_rxn:
            result.append(new_rxn)
    return result


def _deduplicate_reactions(reactions: List[Dict]) -> List[Dict]:
    """Remove duplicate reaction entries (order-preserving)."""
    seen: set = set()
    unique: List[Dict] = []
    for rxn in reactions:
        key = json.dumps(rxn, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(rxn)
    return unique


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_jsonl(input_path: str, output_path: str) -> None:
    transformed: List[Dict] = []

    with open(input_path, encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            image_path = ""
            response_str = "[]"

            try:
                data = json.loads(line, parse_int=str)
                data = _coerce_ints(data)
                image_path  = data.get("images", [{}])[0].get("path", "")
                response_str = data.get("response", "[]") or "[]"
            except (json.JSONDecodeError, ValueError):
                # Salvage with regex
                m = re.search(r'"path":\s*"([^"]+)"', line)
                if m:
                    image_path = m.group(1)
                m = re.search(r'"response":\s*"((?:[^"\\]|\\.)*)"', line)
                if m:
                    response_str = m.group(1).encode("utf-8").decode("unicode_escape")
                print(f"[WARN] Line {line_no}: JSON decode failed; attempting salvage.")

            # Derive relative file_name from the image path
            if image_path:
                parts = image_path.replace("\\", "/").split("/")
                file_name = (os.path.join(parts[-2], parts[-1])
                             if len(parts) >= 2 else os.path.basename(image_path))
            else:
                file_name = ""

            raw = _parse_response(response_str)
            reactions = _transform_reactions(raw)
            reactions = _deduplicate_reactions(reactions)

            transformed.append({"file_name": file_name, "reactions": reactions})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(transformed, fout, indent=2, ensure_ascii=False)

    print(f"Done. {len(transformed)} items written to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert swift-infer JSONL output to evaluation JSON."
    )
    parser.add_argument("--input",  required=True,
                        help="Input JSONL file (swift infer output).")
    parser.add_argument("--output", required=True,
                        help="Output JSON file for evaluation pipeline.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        raise SystemExit(1)

    process_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
