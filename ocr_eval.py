"""Minimal OpenAI file-eval call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
from openai import OpenAI
from PIL import Image, ImageDraw

from ocr_config import (
    DEFAULT_INPUT_DOCUMENT,
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
    input_document: str = DEFAULT_INPUT_DOCUMENT,
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
                input_document,
                manifest["full_detail"],
                manifest["readable_output"],
                manifest["layout_overlay"],
            )
            if path
        ]

        result = run_eval_single(evaluator_model, input_files, prompt)
        result["evaluated_model"] = manifest.get("model", ocr_model)
        result["input_document"] = str(input_document)
        results.append(
            {
                "ocr_model": ocr_model,
                "evaluator_model": evaluator_model,
                "evaluated_model": result["evaluated_model"],
                "input_document": str(input_document),
                "result": result,
            }
        )

    return results


def save_issue_overlays(results, input_document, output_dir):
    colors = {
        "text_accuracy_issue": (220, 53, 69), "structured_data_issue": (0, 123, 255),
        "visual_elements_issue": (111, 66, 193), "layout_reading_order_issue": (255, 140, 0),
        "hallucination_noise_issue": (108, 117, 125),
    }
    input_document, output_dir = Path(input_document), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_pdf = input_document.suffix.lower() == ".pdf"

    for row in results:
        regions = row["result"].get("issue_regions", [])
        pages = sorted({int(r.get("page", 1)) for r in regions if r.get("bbox")})
        if not pages:
            continue
        doc = pymupdf.open(input_document) if is_pdf else None
        try:
            for page_num in pages:
                if is_pdf:
                    if page_num < 1 or page_num > len(doc):
                        continue
                    pix = doc[page_num - 1].get_pixmap(matrix=pymupdf.Matrix(2, 2))
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                else:
                    if page_num != 1:
                        continue
                    image = Image.open(input_document).convert("RGB")
                draw, width, height = ImageDraw.Draw(image), *image.size
                page_regions = [r for r in regions if int(r.get("page", 1)) == page_num]
                for index, region in enumerate(page_regions, start=1):
                    key, bbox = region.get("issue_key"), region.get("bbox", [])
                    if key not in colors or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [max(0, min(1, float(v))) for v in bbox]
                    box = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
                    draw.rectangle(box, outline=colors[key], width=max(3, width // 350))
                    label = f"{index}. {region.get('label') or key.replace('_issue', '').replace('_', ' ')}"
                    y = max(0, box[1] - 18)
                    draw.rectangle([box[0], y, box[0] + len(label) * 7 + 8, y + 18], fill=colors[key])
                    draw.text((box[0] + 4, y + 3), label, fill="white")
                suffix = "" if len(pages) == 1 else f"_page_{page_num}"
                path = output_dir / f"{row['ocr_model']}_issue_overlay{suffix}.png"
                image.save(path)
                print(f"Saved issue overlay: {path}")
        finally:
            if doc:
                doc.close()


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
                "evaluated_model": row.get("evaluated_model", row["ocr_model"]),
                "input_document": row.get("input_document", result["input_document"]),
                "text_accuracy": scores["text_accuracy"],
                "key_value_accuracy": scores["key_value_accuracy"],
                "table_accuracy": scores["table_accuracy"],
                "layout_reading_order": scores["layout_reading_order"],
                "visual_grounding": scores["visual_grounding"],
                "completeness": scores["completeness"],
                "overall": scores["overall"],
                "text_accuracy_issue": result["text_accuracy_issue"],
                "structured_data_issue": result["structured_data_issue"],
                "visual_elements_issue": result["visual_elements_issue"],
                "layout_reading_order_issue": result[
                    "layout_reading_order_issue"
                ],
                "hallucination_noise_issue": result[
                    "hallucination_noise_issue"
                ],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # CLI entry point: OCR model names in, CSV report out.
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr_models", nargs="+")
    parser.add_argument("--input-document", default=DEFAULT_INPUT_DOCUMENT)
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
    results = run_eval_batch(ocr_models, args.input_document, args.model)
    save_eval_csv(results, args.output)
    save_issue_overlays(results, args.input_document, Path(args.output).parent)
    print(f"Saved OCR eval CSV: {args.output}")
