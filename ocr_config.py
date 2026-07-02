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


# ----------------------------- Mistral OCR Options -----------------------------

MISTRAL_CHART_TABLE_ANNOTATION_PROMPT = """Extract visible chart and table content from this document.

Return concise markdown that can be appended to the OCR output. Include chart
titles, axis labels, tick labels, legends, categories, and visible values. Use
markdown tables for chart data or table data when possible. Do not infer values
that are not visible; write "not visible" instead."""

MISTRAL_CHART_TABLE_ANNOTATION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "chart_table_markdown",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": (
                        "Concise markdown containing extracted chart and table "
                        "content. Return an empty string if none is present."
                    ),
                },
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short caveats about missing or unclear values.",
                },
            },
            "required": ["markdown", "notes"],
            "additionalProperties": False,
        },
    },
}


# ----------------------------- Eval Defaults -----------------------------

DEFAULT_INPUT_DOCUMENT = "input/modern_ocr_test.png"
DEFAULT_OCR_EVAL_MODEL = os.getenv("OPENAI_OCR_EVAL_MODEL", "gpt-5.5")
DEFAULT_OCR_EVAL_OUTPUT_DIR = "output/openai_eval"

OCR_EVAL_IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


# ----------------------------- Eval Manifests -----------------------------

OCR_EVAL_MANIFESTS = {
    "paddleocr": {
        "model": "PaddleOCR PPStructureV3",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/paddleocr/full_detail.json",
        "readable_output": "output/paddleocr/readable_output.md",
        "layout_overlay": "output/paddleocr/layout_overlay.png",
    },
    "nuextract3": {
        "model": "NuExtract 3",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/nuextract3/full_detail.txt",
        "readable_output": "output/nuextract3/readable_output.txt",
        "layout_overlay": None,
    },
    "chandra_ocr_2": {
        "model": "Chandra OCR 2",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/chandra_ocr_2/full_detail.json",
        "readable_output": "output/chandra_ocr_2/readable_output.md",
        "layout_overlay": "output/chandra_ocr_2/layout_overlay.png",
    },
    "landingai": {
        "model": "LandingAI DPT-2",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/landingai/full_detail.json",
        "readable_output": "output/landingai/readable_output.md",
        "layout_overlay": "output/landingai/layout_overlay.png",
    },
    "llamacloud": {
        "model": "LlamaParse",
        "input_document": DEFAULT_INPUT_DOCUMENT,
        "full_detail": "output/llamacloud/full_detail.json",
        "readable_output": "output/llamacloud/readable_output.md",
        "layout_overlay": "output/llamacloud/layout_overlay.png",
    },
    "mistralocr": {
        "model": "Mistral OCR",
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
regions. Treat information as present if it appears in any provided OCR text
artifact, including full-detail fields, readable output, or structured
annotations. Do not mark content as missing only because it appears in full
detail rather than readable output; instead, mention that as a readability or
usability issue if it matters. For form controls, if checkbox or radio-button
states appear in any provided artifact, such as full-detail HTML or structured
annotations, treat those states as captured. Do not report them as missing,
incorrect, or as a readability/usability issue solely because the readable
Markdown flattens or omits the control markup.

Do not penalize provider markup that exists to preserve structure, grounding, or
semantic visual descriptions. Examples include chunk anchors like
<a id='...'></a>, table cell IDs, and pseudo-tags such as <::visual content::>,
<::chart::>, or <::attestation::>. Treat these as acceptable output format
markers unless they obscure, replace, or contradict the document content.

For charts converted to tables, judge whether the semantic chart data is
preserved. If the chart title, category labels, values, and units are captured,
do not penalize missing axis ticks, gridlines, or other visual scaffolding. Do
not call values like "98" missing percent signs when the table/header already
states the unit, such as "Accuracy (%)". Generic inferred headers are acceptable
when they faithfully describe the chart axes or categories. A blank category
column header is not an issue when the chart title or nearby text clearly
defines that category, such as "BY FONT TYPE". For these harmless formatting
quirks, return "No significant issue." rather than creating a minor issue.
Treat isolated raw OCR noise as minor if it is not present in the readable or
structured output and does not change what a user consumes. If no layout overlay
is configured, say that visual grounding cannot be verified from an overlay and
reflect that in the visual_grounding score.

Return one concise human-review finding for each issue category:
text_accuracy_issue, structured_data_issue, visual_elements_issue,
layout_reading_order_issue, and hallucination_noise_issue. Each finding must be
one self-contained sentence. If a category has no meaningful issue, write
"No significant issue.".

Also return issue_regions with at most one localizable region per issue category.
Use 1-based page numbers and normalized [x1, y1, x2, y2] boxes relative to the
original page. Leave abstract or non-localizable issues out of issue_regions."""

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
        "text_accuracy_issue": {"type": "string"},
        "structured_data_issue": {"type": "string"},
        "visual_elements_issue": {"type": "string"},
        "layout_reading_order_issue": {"type": "string"},
        "hallucination_noise_issue": {"type": "string"},
        "issue_regions": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "enum": [
                            "text_accuracy_issue",
                            "structured_data_issue",
                            "visual_elements_issue",
                            "layout_reading_order_issue",
                            "hallucination_noise_issue",
                        ],
                    },
                    "page": {"type": "integer"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string"},
                },
                "required": ["issue_key", "page", "bbox", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "evaluated_model",
        "input_document",
        "scores",
        "text_accuracy_issue",
        "structured_data_issue",
        "visual_elements_issue",
        "layout_reading_order_issue",
        "hallucination_noise_issue",
        "issue_regions",
    ],
    "additionalProperties": False,
}
