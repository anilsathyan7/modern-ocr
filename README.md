# Modern OCR


Modern OCR systems are no longer just text recognizers; they are document
understanding systems. A useful OCR pipeline should preserve the document's
text, structure, visual grounding, and meaning in a form that downstream tools
can use and humans can review.

This project compares recent open-source and cloud OCR systems on the same set
of documents, then evaluates how well each one preserves the document's text,
structure, reading order, visual grounding, and structured data.

## OCR Features

- **Text recognition**: Handles different fonts, sizes, languages, orientations,
  and document quality levels.
- **Reading order**: Preserves the natural flow of multi-column and complex
  documents.
- **Layout understanding**: Detects titles, headings, paragraphs, lists,
  captions, headers, footers, marginalia, and page numbers.
- **Tables and forms**: Preserves rows, columns, merged cells, form fields,
  checkboxes, radio buttons, and selected marks.
- **Key-value extraction**: Extracts structured fields from invoices, receipts,
  IDs, certificates, forms, and similar documents.
- **Charts and figures**: Extracts chart titles, labels, legends, values, units,
  figures, images, logos, and visual meaning.
- **Handwriting and symbols**: Handles handwritten text, formulas, equations,
  and scientific notation where supported.
- **Seals and attestations**: Detects stamps, seals, signatures, attestations,
  and official document markings.
- **Visual grounding**: Returns bounding boxes, coordinates, layout regions,
  confidence scores, or overlays linked to the source page.
- **Flexible input and output**: Handles PDFs, images, scans, Word-style
  documents, and outputs Markdown, HTML, JSON, or structured text.
- **Custom extraction**: Supports schema-based extraction, custom prompts,
  document classification, and task-specific parsing.
- **Robustness**: Handles low-quality scans, blur, skew, rotation, shadows,
  compression artifacts, multi-page documents, and repeated headers or footers.

## Requirements

- Linux x86_64
- Python 3.10 to 3.13
- `uv`
- CUDA-compatible GPU setup for `paddlepaddle-gpu==3.3.1`

## Install

Install the project dependencies:

```bash
uv sync
```

## API Keys

Set only the keys for the OCR providers you plan to run:

```bash
export OPENAI_API_KEY="..."
export LLAMA_CLOUD_API_KEY="..."
export MISTRAL_API_KEY="..."
export VISION_AGENT_API_KEY="..."
```

## Run OCR

Generate OCR outputs for the configured parsers. If no document path is passed,
`ocr_models.py` uses `input/document.png`.

```bash
# default input document
uv run python ocr_models.py

# custom input document
uv run python ocr_models.py input/virology_pg2.pdf
```

Outputs are written under `output/<parser_name>/`, for example
`output/paddleocr/readable_output.md` and `output/paddleocr/full_detail.json`.

## Run Evaluation

Compare saved OCR outputs against the original document and write the scores to
CSV:

```bash
# all outputs
uv run python ocr_eval.py all

# selected outputs
uv run python ocr_eval.py paddleocr llamacloud mistralocr

# custom CSV path
uv run python ocr_eval.py all --output output/openai_eval/results.csv
```

The evaluation CSV is written to `output/openai_eval/results.csv` by default.
Issue overlay images are also written to the same directory when the evaluator
returns localizable issue regions.
