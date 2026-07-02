from __future__ import annotations

import argparse
import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from chandra.input import load_file
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from llama_cloud import LlamaCloud
from mistralai.client import Mistral
from paddleocr import PPStructureV3
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from landingai_ade import LandingAIADE
from ocr_config import (
    MISTRAL_CHART_TABLE_ANNOTATION_FORMAT,
    MISTRAL_CHART_TABLE_ANNOTATION_PROMPT,
)
from ocr_helper import (
    ensure_output_dir,
    flush_gpu_memory,
    pdf_to_images,
    save_chandra_outputs,
    save_landingai_outputs,
    save_llamacloud_outputs,
    save_mistralocr_outputs,
    save_nuextract3_outputs,
    save_paddleocr_outputs,
)


class OCRModelBase(ABC):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False

    @abstractmethod
    def predict(self, document_path, save_to="./output"):
        """Run OCR/parser model and save outputs."""

    def release(self):
        """Release model-owned references and cached GPU memory."""
        flush_gpu_memory()


class PPOCRDocParser(OCRModelBase):
    def __init__(
        self,
        output_prefix="paddleocr",
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        markdown_ignore_labels=None,
    ):
        self.output_prefix = output_prefix
        if markdown_ignore_labels is None:
            markdown_ignore_labels = []

        # Configure PaddleOCR structure pipeline.
        self.pipeline = PPStructureV3(
            lang="eng",
            device="gpu",
            layout_detection_model_name="PP-DocLayout_plus-L",
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="PP-OCRv6_medium_rec",
            # Document optimization parameters
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            format_block_content=True,
            markdown_ignore_labels=markdown_ignore_labels,
            # Advanced modular extraction
            use_formula_recognition=True,
            use_table_recognition=True,
            use_chart_recognition=True,
            use_seal_recognition=True,
            use_region_detection=True,
            use_textline_orientation=True,
        )

    def predict(
        self,
        image_path,
        save_to="./output",
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        markdown_ignore_labels=None,
    ):
        output_dir = ensure_output_dir(save_to, self.output_prefix)

        paddle_options = {
            "layout_threshold": layout_threshold,
            "layout_nms": layout_nms,
            "layout_unclip_ratio": layout_unclip_ratio,
            "layout_merge_bboxes_mode": layout_merge_bboxes_mode,
            "markdown_ignore_labels": markdown_ignore_labels,
        }
        paddle_options = {
            key: value for key, value in paddle_options.items() if value is not None
        }

        # Parse document structure.
        results = self.pipeline.predict(
            input=image_path,
            **paddle_options,
        )

        return save_paddleocr_outputs(
            results,
            save_to=output_dir,
        )

    def release(self):
        self.pipeline = None
        super().release()


class NuExtract3DocParser(OCRModelBase):
    def __init__(
        self,
        model_id="numind/NuExtract3",
        device_map="auto",
        max_new_tokens=4096,
        output_prefix="nuextract3",
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.output_prefix = output_prefix

        # Load processor and model.
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

    def predict(
        self,
        image_path,
        mode="content",
        enable_thinking=False,
        save_to="./output",
    ):
        output_dir = ensure_output_dir(save_to, self.output_prefix)

        # Load input image.
        image_path = Path(image_path)
        if image_path.suffix.lower() == ".pdf":
            image_path = pdf_to_images(image_path)[0]

        with Image.open(image_path) as image:
            document_image = image.convert("RGB")

        # Build model inputs.
        inputs = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": document_image,
                        }
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            mode=mode,
            enable_thinking=enable_thinking,
        ).to(self.model.device)

        # Generate OCR text.
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
        )

        # Decode only new tokens.
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return save_nuextract3_outputs(output, save_to=output_dir)

    def release(self):
        self.model = None
        self.processor = None
        super().release()


class ChandraOCR2DocParser(OCRModelBase):
    def __init__(
        self,
        model_id="datalab-to/chandra-ocr-2",
        device_map="auto",
        prompt_type="ocr_layout",
        output_prefix="chandra_ocr_2",
    ):
        self.model_id = model_id
        self.prompt_type = prompt_type
        self.output_prefix = output_prefix

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
        self.model.processor = AutoProcessor.from_pretrained(model_id)
        self.model.processor.tokenizer.padding_side = "left"

    def predict(
        self,
        document_path,
        save_to="./output",
        include_headers_footers=True,
        include_images=True,
        draw_boxes=True,
        page_range=None,
    ):
        output_dir = ensure_output_dir(save_to, self.output_prefix)
        config = {"page_range": page_range} if page_range else {}
        images = load_file(str(document_path), config)
        batch = [
            BatchInputItem(
                image=image,
                prompt_type=self.prompt_type,
            )
            for image in images
        ]

        results = generate_hf(batch, self.model)
        return save_chandra_outputs(
            results,
            images,
            save_to=output_dir,
            include_headers_footers=include_headers_footers,
            include_images=include_images,
            draw_boxes=draw_boxes,
        )

    def release(self):
        self.model = None
        super().release()


class LandingAIDocParser(OCRModelBase):
    def __init__(
        self,
        model="dpt-2-latest",
        api_key=None,
        output_prefix="landingai",
    ):
        api_key = api_key if api_key else os.getenv("VISION_AGENT_API_KEY")
        # Create LandingAI client.
        self.client = LandingAIADE(apikey=api_key)
        self.model = model
        self.output_prefix = output_prefix

    def predict(
        self,
        document_path,
        save_to="./output",
        draw_boxes=True,
        save_chunks=True,
    ) -> Any:
        # Prepare output path.
        document_path = Path(document_path)
        output_dir = ensure_output_dir(save_to, self.output_prefix)

        # Parse with LandingAI.
        parse_response = self.client.parse(
            document=document_path,
            model=self.model,
        )

        return save_landingai_outputs(
            parse_response,
            document_path,
            save_to=output_dir,
            draw_boxes=draw_boxes,
            save_chunks=save_chunks,
        )


class LlamaCloudDocParser(OCRModelBase):
    def __init__(
        self,
        tier="agentic",
        version="latest",
        api_key=None,
        output_prefix="llamacloud",
    ):
        api_key = api_key if api_key else os.getenv("LLAMA_CLOUD_API_KEY")
        # Create LlamaCloud client.
        self.client = LlamaCloud(api_key=api_key)
        self.tier = tier
        self.version = version
        self.output_prefix = output_prefix

    def predict(self, document_path, save_to="./output") -> Any:
        output_dir = ensure_output_dir(save_to, self.output_prefix)

        # Upload document for parsing.
        file_obj = self.client.files.create(
            file=document_path,
            purpose="parse",
        )
        # Parse with LlamaCloud.
        result = self.client.parsing.parse(
            file_id=file_obj.id,
            tier=self.tier,
            version=self.version,
            expand=[
                "markdown_full",
                "text_full",
                "items",
                "metadata",
                "job_metadata",
                "items_content_metadata",
                "images_content_metadata",
                "raw_words_content_metadata",
            ],
            output_options={
                "granular_bboxes": ["word", "line", "cell"],
                "images_to_save": ["layout"],
                "spatial_text": {
                    "preserve_very_small_text": True,
                },
            },
            processing_options={
                "cost_optimizer": {"enable": False},
            },
        )

        return save_llamacloud_outputs(
            result,
            document_path=document_path,
            save_to=output_dir,
        )


class MistralOCRDocParser(OCRModelBase):
    def __init__(
        self,
        model="mistral-ocr-latest",
        api_key=None,
        output_prefix="mistralocr",
    ):
        api_key = api_key if api_key else os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.output_prefix = output_prefix

    def predict(
        self,
        document_path,
        save_to="./output",
        pages=None,
        include_blocks=True,
        include_image_base64=True,
        confidence_scores_granularity="word",
        table_format=None,
        extract_header=False,
        extract_footer=False,
        extract_chart_table_markdown=True,
        draw_boxes=True,
        save_images=True,
    ) -> Any:
        document_path = Path(document_path)
        output_dir = ensure_output_dir(save_to, self.output_prefix)

        with document_path.open("rb") as file:
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": document_path.name,
                    "content": file,
                },
                purpose="ocr",
            )

        signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
        ocr_options = {
            "pages": pages,
            "include_blocks": include_blocks,
            "include_image_base64": include_image_base64,
            "confidence_scores_granularity": confidence_scores_granularity,
            "table_format": table_format,
            "extract_header": extract_header,
            "extract_footer": extract_footer,
        }
        if extract_chart_table_markdown:
            ocr_options.update(
                {
                    "document_annotation_format": (
                        MISTRAL_CHART_TABLE_ANNOTATION_FORMAT
                    ),
                    "document_annotation_prompt": (
                        MISTRAL_CHART_TABLE_ANNOTATION_PROMPT
                    ),
                }
            )
        ocr_options = {
            key: value for key, value in ocr_options.items() if value is not None
        }

        result = self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            **ocr_options,
        )

        return save_mistralocr_outputs(
            result,
            document_path=document_path,
            save_to=output_dir,
            draw_boxes=draw_boxes,
            save_images=save_images,
        )

    def release(self):
        self.client = None
        super().release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "document_path",
        nargs="?",
        default="input/document.png",
        help="Document/image to parse.",
    )
    args = parser.parse_args()
    document_path = args.document_path

    with PPOCRDocParser() as model:
        output = model.predict(document_path)

    with NuExtract3DocParser() as model:
        output = model.predict(document_path)
        print(output)

    with ChandraOCR2DocParser() as model:
        output = model.predict(document_path)
        print(output)

    with LandingAIDocParser() as model:
        output = model.predict(document_path)

    with LlamaCloudDocParser() as model:
        output = model.predict(document_path)
        print(output.job.status)

    with MistralOCRDocParser() as model:
        output = model.predict(document_path)
        print(output.usage_info)
