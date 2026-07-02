# OCR

Setup for the OCR comparison/evaluation repo.

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
