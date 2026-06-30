"""Minimal OpenAI file-eval call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from ocr_config import (
    DEFAULT_OCR_EVAL_MODEL,
    DEFAULT_OCR_EVAL_OUTPUT_DIR,
    OCR_EVAL_IMAGE_EXTENSIONS,
    OCR_EVAL_MANIFESTS,
    OCR_EVAL_SCHEMA,
    OCR_EVAL_SYSTEM_PROMPT,
    OCR_EVAL_USER_PROMPT,
)


def run_eval_single(
    model: str,
    input_files: list[str],
    prompt: str = OCR_EVAL_SYSTEM_PROMPT,
) -> dict[str, Any]:
    client = OpenAI()
    content = [{"type": "input_text", "text": OCR_EVAL_USER_PROMPT}]

    # Upload each configured file and attach it with the right input type.
    for input_file in input_files:
        path = Path(input_file)

        is_image = path.suffix.lower() in OCR_EVAL_IMAGE_EXTENSIONS
        purpose = "vision" if is_image else "user_data"
        content_type = "input_image" if is_image else "input_file"

        with path.open("rb") as file:
            uploaded_file = client.files.create(file=file, purpose=purpose)

        content.append({"type": "input_text", "text": f"File: {path.name}"})
        content.append({"type": content_type, "file_id": uploaded_file.id})

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ocr_eval_result",
                "schema": OCR_EVAL_SCHEMA,
                "strict": True,
            }
        },
    )
    result = json.loads(response.output_text)
    return result


# Run one structured eval per configured OCR model.
def run_eval_batch(
    ocr_models: list[str],
    evaluator_model: str = DEFAULT_OCR_EVAL_MODEL,
    prompt: str = OCR_EVAL_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    results = []

    for ocr_model in ocr_models:
        manifest = OCR_EVAL_MANIFESTS[ocr_model]
        print(f"Evaluating OCR model: {ocr_model}")
        input_files = [
            path
            for path in (
                manifest["input_document"],
                manifest["full_detail"],
                manifest["readable_output"],
                manifest["layout_overlay"],
            )
            if path
        ]

        result = run_eval_single(evaluator_model, input_files, prompt)
        results.append(
            {
                "ocr_model": ocr_model,
                "evaluator_model": evaluator_model,
                "result": result,
            }
        )

    return results


# Flatten schema output into score-focused CSV rows.
def save_eval_csv(results: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in results:
        result = row["result"]
        scores = result["scores"]
        rows.append(
            {
                "ocr_model": row["ocr_model"],
                "evaluator_model": row["evaluator_model"],
                "evaluated_model": result["evaluated_model"],
                "input_document": result["input_document"],
                "text_accuracy": scores["text_accuracy"],
                "key_value_accuracy": scores["key_value_accuracy"],
                "table_accuracy": scores["table_accuracy"],
                "layout_reading_order": scores["layout_reading_order"],
                "visual_grounding": scores["visual_grounding"],
                "completeness": scores["completeness"],
                "overall": scores["overall"],
                "summary": result["summary"],
                "strengths": json.dumps(result["strengths"]),
                "major_misses": json.dumps(result["major_misses"]),
                "layout_notes": json.dumps(result["layout_notes"]),
                "recommended_next_checks": json.dumps(
                    result["recommended_next_checks"]
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # CLI entry point: OCR model names in, CSV report out.
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr_models", nargs="+")
    parser.add_argument("--model", default=DEFAULT_OCR_EVAL_MODEL)
    parser.add_argument(
        "--output",
        default=Path(DEFAULT_OCR_EVAL_OUTPUT_DIR) / "results.csv",
    )
    args = parser.parse_args()

    # select ocr models to evaluate
    ocr_models = (
        list(OCR_EVAL_MANIFESTS)
        if args.ocr_models == ["all"]
        else args.ocr_models
    )

    # run the evaluation and save results
    results = run_eval_batch(ocr_models, args.model)
    save_eval_csv(results, args.output)
    print(f"Saved OCR eval CSV: {args.output}")
