import os


# ----------------------------- Overlay Rendering -----------------------------

CHUNK_TYPE_COLORS = {
    "chunkText": (40, 167, 69),
    "chunkTable": (0, 123, 255),
    "chunkMarginalia": (111, 66, 193),
    "chunkFigure": (255, 0, 255),
    "chunkLogo": (144, 238, 144),
    "chunkCard": (255, 165, 0),
    "chunkAttestation": (0, 255, 255),
    "chunkScanCode": (255, 193, 7),
    "chunkForm": (220, 20, 60),
    "tableCell": (173, 216, 230),
    "table": (70, 130, 180),
}

LLAMA_ITEM_TYPE_COLORS = {
    "heading": (0, 123, 255),
    "text": (40, 167, 69),
    "table": (220, 20, 60),
    "image": (255, 165, 0),
    "chart": (111, 66, 193),
}

CHANDRA_LABEL_COLORS = {
    "Caption": (111, 66, 193),
    "Figure": (255, 165, 0),
    "Footnote": (108, 117, 125),
    "Formula": (102, 16, 242),
    "Image": (255, 165, 0),
    "List-Group": (23, 162, 184),
    "Page-Footer": (108, 117, 125),
    "Page-Header": (108, 117, 125),
    "Section-Header": (0, 123, 255),
    "Table": (220, 20, 60),
    "Text": (40, 167, 69),
}

PADDLEOCR_LABEL_COLORS = {
    "chart": (111, 66, 193),
    "figure": (255, 165, 0),
    "figure_title": (0, 123, 255),
    "footer": (108, 117, 125),
    "header": (0, 123, 255),
    "image": (255, 165, 0),
    "number": (220, 53, 69),
    "paragraph_title": (0, 123, 255),
    "table": (220, 20, 60),
    "text": (40, 167, 69),
}

MISTRALOCR_LABEL_COLORS = {
    "caption": (111, 66, 193),
    "figure": (255, 165, 0),
    "footer": (108, 117, 125),
    "header": (108, 117, 125),
    "heading": (0, 123, 255),
    "image": (255, 165, 0),
    "list": (23, 162, 184),
    "paragraph": (40, 167, 69),
    "table": (220, 20, 60),
    "text": (40, 167, 69),
    "title": (0, 123, 255),
}

PDF_RENDER_DPI = 300


# ----------------------------- Eval Defaults -----------------------------

DEFAULT_INPUT_DOCUMENT = "input/utility_bill.pdf"
DEFAULT_OCR_EVAL_MODEL = os.getenv("OPENAI_OCR_EVAL_MODEL", "gpt-5.5")
DEFAULT_OCR_EVAL_OUTPUT_DIR = "output/openai_eval"

OCR_EVAL_IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


# ----------------------------- Eval Manifests -----------------------------

OCR_EVAL_MANIFESTS = {
    "paddleocr": {
        "model": "paddleocr",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/paddleocr/full_detail.json",
        "readable_output": "output/paddleocr/readable_output.md",
        "layout_overlay": "output/paddleocr/layout_overlay.png",
    },
    "nuextract3": {
        "model": "nuextract3",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/nuextract3/full_detail.txt",
        "readable_output": "output/nuextract3/readable_output.txt",
        "layout_overlay": None,
    },
    "chandra_ocr_2": {
        "model": "chandra_ocr_2",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/chandra_ocr_2/full_detail.json",
        "readable_output": "output/chandra_ocr_2/readable_output.md",
        "layout_overlay": "output/chandra_ocr_2/layout_overlay.png",
    },
    "landingai": {
        "model": "landingai",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/landingai/full_detail.json",
        "readable_output": "output/landingai/readable_output.md",
        "layout_overlay": "output/landingai/layout_overlay.png",
    },
    "llamacloud": {
        "model": "llamacloud",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/llamacloud/full_detail.json",
        "readable_output": "output/llamacloud/readable_output.md",
        "layout_overlay": "output/llamacloud/layout_overlay.png",
    },
    "mistralocr": {
        "model": "mistralocr",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/mistralocr/full_detail.json",
        "readable_output": "output/mistralocr/readable_output.md",
        "layout_overlay": "output/mistralocr/layout_overlay.png",
    },
}


# ----------------------------- Eval Prompt -----------------------------

OCR_EVAL_SYSTEM_PROMPT = """You are a strict OCR and document-layout evaluation judge.

Use the original input document as the source of truth. Evaluate whether the OCR
model output captures the full document: all visible text, small footer/payment
regions, key-value fields, tables, chart labels, reading order, and layout
grounding. Use the full-detail file for raw OCR/layout evidence, the readable
output for what a user would consume, and the overlay image, when provided, to
judge region detection quality.

Scores must be 0-100, where 100 means effectively complete and faithful. Penalize
missing content, hallucinated content, wrong values, broken table structure,
incorrect reading order, and layout overlays that miss or misclassify important
regions. If no layout overlay is configured, say that visual grounding cannot be
verified from an overlay and reflect that in the visual_grounding score. Be
concrete and cite the provided file roles in evidence_files."""

OCR_EVAL_USER_PROMPT = "Evaluate the attached OCR files."


# ----------------------------- Eval Schema -----------------------------

OCR_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluated_model": {"type": "string"},
        "input_document": {"type": "string"},
        "scores": {
            "type": "object",
            "properties": {
                "text_accuracy": {"type": "number"},
                "key_value_accuracy": {"type": "number"},
                "table_accuracy": {"type": "number"},
                "layout_reading_order": {"type": "number"},
                "visual_grounding": {"type": "number"},
                "completeness": {"type": "number"},
                "overall": {"type": "number"},
            },
            "required": [
                "text_accuracy",
                "key_value_accuracy",
                "table_accuracy",
                "layout_reading_order",
                "visual_grounding",
                "completeness",
                "overall",
            ],
            "additionalProperties": False,
        },
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "major_misses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "major", "minor"],
                    },
                    "finding": {"type": "string"},
                    "expected": {"type": "string"},
                    "observed": {"type": "string"},
                    "evidence_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "severity",
                    "finding",
                    "expected",
                    "observed",
                    "evidence_files",
                ],
                "additionalProperties": False,
            },
        },
        "layout_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommended_next_checks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "evaluated_model",
        "input_document",
        "scores",
        "summary",
        "strengths",
        "major_misses",
        "layout_notes",
        "recommended_next_checks",
    ],
    "additionalProperties": False,
}
