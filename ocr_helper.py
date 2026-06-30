import gc
import json
import tempfile

from base64 import b64decode
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pymupdf
import paddle
import torch
from chandra.output import extract_images, parse_chunks, parse_html, parse_markdown
from PIL import Image, ImageDraw

from ocr_config import (
    CHANDRA_LABEL_COLORS,
    CHUNK_TYPE_COLORS,
    LLAMA_ITEM_TYPE_COLORS,
    MISTRALOCR_LABEL_COLORS,
    PADDLEOCR_LABEL_COLORS,
    PDF_RENDER_DPI,
)


# ----------------------------- Shared Utilities -----------------------------

def ensure_output_dir(save_to, output_prefix):
    output_dir = Path(save_to) / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def flush_gpu_memory():
    """Release cached GPU memory between parser runs."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()

    gc.collect()


def pdf_to_images(pdf_path, dpi=PDF_RENDER_DPI, output_dir=None):
    """Render PDF pages to PNG files at OCR-friendly resolution."""
    pdf_path = Path(pdf_path)
    output_dir = (
        Path(tempfile.mkdtemp(prefix="pdf_ocr_"))
        if output_dir is None
        else Path(output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    image_paths = []

    try:
        for page_index, page in enumerate(doc, start=1):
            output_path = output_dir / f"page_{page_index:04d}.png"
            zoom = dpi / 72
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom),
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            pix.save(output_path)
            image_paths.append(output_path)
    finally:
        doc.close()

    return image_paths


# ----------------------------- LandingAI Helpers -----------------------------

def print_landingai_parse_summary(parse_response):
    """Print a short summary of a Landing AI parse response."""
    print("Parsing completed.")
    print(f"job_id: {getattr(parse_response.metadata, 'job_id', None)}")
    print(f"Total pages: {len(parse_response.splits)}")
    print(
        f"Total time (ms): "
        f"{getattr(parse_response.metadata, 'duration_ms', None)}"
    )
    print(f"Total markdown characters: {len(parse_response.markdown)}")
    print(f"Number of chunks: {len(parse_response.chunks)}")
    print(" ")
    print("Complete Markdown:")
    print(parse_response.markdown)


def save_landingai_layout_overlay(
    parse_response,
    document_path,
    save_to="./output",
    dpi=PDF_RENDER_DPI,
):
    """Save LandingAI chunk layout overlay images."""

    def create_annotated_image(image, groundings, page_num=0):
        """Create an annotated image with grounding boxes and labels."""
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)

        img_width, img_height = image.size

        for gid, grounding in groundings.items():
            if grounding.page != page_num:
                continue

            box = grounding.box
            left, top, right, bottom = box.left, box.top, box.right, box.bottom

            x1 = int(left * img_width)
            y1 = int(top * img_height)
            x2 = int(right * img_width)
            y2 = int(bottom * img_height)

            color = CHUNK_TYPE_COLORS.get(grounding.type, (128, 128, 128))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label = f"{grounding.type}:{gid}"
            label_y = max(0, y1 - 20)
            draw.rectangle([x1, label_y, x1 + len(label) * 8, y1], fill=color)
            draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))

        return annotated_img

    document_path = Path(document_path)
    save_dir = Path(save_to)
    save_dir.mkdir(parents=True, exist_ok=True)

    if document_path.suffix.lower() == ".pdf":
        image_paths = pdf_to_images(document_path, dpi=dpi)
        for page_num, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                img = image.copy()

            annotated_img = create_annotated_image(
                img,
                parse_response.grounding,
                page_num,
            )
            annotated_path = (
                save_dir / "layout_overlay.png"
                if len(image_paths) == 1
                else save_dir / f"layout_overlay_page_{page_num + 1}.png"
            )
            annotated_img.save(annotated_path)
            print(f"Annotated image saved to: {annotated_path}")
    else:
        img = Image.open(document_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        annotated_img = create_annotated_image(img, parse_response.grounding)
        annotated_path = save_dir / "layout_overlay.png"
        annotated_img.save(annotated_path)
        print(f"Annotated image saved to: {annotated_path}")

    return None


def save_landingai_chunks(
    parse_response,
    document_path,
    output_base_dir="./output",
    dpi=PDF_RENDER_DPI,
):
    """Save each parsed chunk as a separate image file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    document_path = Path(document_path)
    output_dir = Path(output_base_dir) / "groundings" / f"{document_path.stem}_{timestamp}"

    def save_page_chunks(image, chunks, page_num):
        """Save all chunks for a specific page."""
        img_width, img_height = image.size

        page_dir = output_dir / f"page_{page_num}"
        page_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            if chunk.grounding.page != page_num:
                continue

            box = chunk.grounding.box

            x1 = int(box.left * img_width)
            y1 = int(box.top * img_height)
            x2 = int(box.right * img_width)
            y2 = int(box.bottom * img_height)

            chunk_img = image.crop((x1, y1, x2, y2))

            filename = f"{chunk.type}.{chunk.id}.png"
            output_path = page_dir / filename
            chunk_img.save(output_path)

            print(f"Saved chunk: {output_path}")

    if document_path.suffix.lower() == ".pdf":
        for page_num, image_path in enumerate(pdf_to_images(document_path, dpi=dpi)):
            with Image.open(image_path) as image:
                img = image.copy()
            save_page_chunks(img, parse_response.chunks, page_num)
    else:
        img = Image.open(document_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        save_page_chunks(img, parse_response.chunks, 0)

    print(f"\nAll chunks saved to: {output_dir}")
    return output_dir


# ----------------------------- PaddleOCR Helpers -----------------------------

def save_paddleocr_outputs(
    results,
    save_to="./output",
    output_name="output",
):
    """Save PaddleOCR markdown, JSON, annotated images, and Word outputs."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    for page_index, result in enumerate(results, start=1):
        result["input_path"] = f"{output_name}.pdf"
        result.save_to_markdown(save_path=str(save_to / "readable_output.md"))
        result.save_to_json(save_path=str(save_to / "full_detail.json"))
        result.save_to_img(save_path=str(save_to))
        result.save_to_word(save_path=str(save_to))
        save_paddleocr_layout_overlay(
            result,
            save_to=save_to,
            page_index=page_index,
            page_count=len(results),
        )

    return results


def save_paddleocr_layout_overlay(
    result,
    save_to="./output",
    page_index=1,
    page_count=1,
):
    """Save PaddleOCR layout overlay images with consistent filenames."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    doc_preprocessor_res = result["doc_preprocessor_res"]
    base_image = Image.fromarray(doc_preprocessor_res["output_img"][:, :, ::-1])
    blocks = result["parsing_res_list"]
    images = result.img

    if base_image is None or not blocks:
        if "layout_det_res" not in images:
            return None
        annotated_image = images["layout_det_res"]
    else:
        annotated_image = base_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for block_index, block in enumerate(blocks, start=1):
            bbox = block.bbox
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = block.label
            color = PADDLEOCR_LABEL_COLORS.get(label, (128, 128, 128))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            tag = f"{block_index}:{label}"
            label_y = max(0, y1 - 18)
            draw.rectangle([x1, label_y, x1 + len(tag) * 7, y1], fill=color)
            draw.text((x1 + 2, label_y + 2), tag, fill=(255, 255, 255))

    if page_count == 1:
        output_path = save_to / "layout_overlay.png"
    else:
        output_path = save_to / f"layout_overlay_page_{page_index}.png"

    annotated_image.save(output_path)

    if "layout_order_res" in images:
        if page_count == 1:
            order_output_path = save_to / "output_layout_order.png"
        else:
            order_output_path = save_to / f"output_page_{page_index}_layout_order.png"
        images["layout_order_res"].save(order_output_path)

    print(f"Saved PaddleOCR bbox image: {output_path}")
    return output_path


# ----------------------------- NuExtract3 Helpers -----------------------------

def save_nuextract3_outputs(output, save_to="./output"):
    """Save NuExtract3 text output."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)
    (save_to / "full_detail.txt").write_text(output, encoding="utf-8")
    (save_to / "readable_output.txt").write_text(output, encoding="utf-8")
    return output


# ----------------------------- Chandra Helpers -----------------------------

def save_chandra_outputs(
    results,
    images,
    save_to="./output",
    include_headers_footers=False,
    include_images=True,
    draw_boxes=True,
):
    """Save Chandra OCR markdown, HTML, chunks, and extracted images."""
    output_kwargs = {
        "include_headers_footers": include_headers_footers,
        "include_images": include_images,
    }

    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    page_markdowns = []
    page_html = []
    page_metadata = []

    for page_num, (result, image) in enumerate(zip(results, images), start=1):
        raw_html = result.raw
        markdown = parse_markdown(raw_html, **output_kwargs).strip()
        html = parse_html(raw_html, **output_kwargs).strip()
        chunks = parse_chunks(raw_html, image)
        extracted_images = (
            extract_images(raw_html, chunks, image)
            if include_images
            else {}
        )

        page_markdowns.append(markdown)
        page_html.append(html)
        page_metadata.append(
            {
                "page_num": page_num,
                "page_box": [0, 0, image.width, image.height],
                "token_count": result.token_count,
                "num_chunks": len(chunks),
                "num_images": len(extracted_images),
                "chunks": chunks,
            }
        )

        for image_name, extracted_image in extracted_images.items():
            extracted_image.save(save_to / image_name)

    markdown = "\n\n---\n\n".join(page_markdowns)
    html = "\n\n".join(page_html)
    chunks_metadata = {
        "num_pages": len(page_metadata),
        "pages": page_metadata,
    }

    (save_to / "readable_output.md").write_text(markdown, encoding="utf-8")
    (save_to / "output.html").write_text(html, encoding="utf-8")
    (save_to / "full_detail.json").write_text(
        json.dumps(chunks_metadata, indent=2),
        encoding="utf-8",
    )
    if draw_boxes:
        save_chandra_layout_overlay(
            images,
            chunks_metadata,
            save_to=save_to,
        )

    return markdown


def save_chandra_layout_overlay(
    images,
    chunks_metadata,
    save_to="./output",
):
    """Draw Chandra layout chunk boxes on each page image."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    output_paths = []
    pages = chunks_metadata.get("pages", [])

    for page_index, (image, page) in enumerate(zip(images, pages), start=1):
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for chunk_index, chunk in enumerate(page.get("chunks", []), start=1):
            bbox = chunk.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = chunk.get("label", "Block")
            color = CHANDRA_LABEL_COLORS.get(label, (128, 128, 128))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            tag = f"{chunk_index}:{label}"
            label_y = max(0, y1 - 18)
            draw.rectangle([x1, label_y, x1 + len(tag) * 7, y1], fill=color)
            draw.text((x1 + 2, label_y + 2), tag, fill=(255, 255, 255))

        if len(pages) == 1:
            output_path = save_to / "layout_overlay.png"
        else:
            output_path = save_to / f"layout_overlay_page_{page_index}.png"

        annotated_image.save(output_path)
        print(f"Saved Chandra bbox image: {output_path}")
        output_paths.append(output_path)

    return output_paths


# ----------------------------- LandingAI Outputs -----------------------------

def save_landingai_outputs(
    parse_response,
    document_path,
    save_to="./output",
    draw_boxes=True,
    save_chunks=True,
):
    """Save LandingAI parse outputs and optional visual grounding artifacts."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    print_landingai_parse_summary(parse_response)

    if draw_boxes:
        save_landingai_layout_overlay(
            parse_response,
            document_path,
            save_to=save_to,
        )

    if save_chunks:
        save_landingai_chunks(
            parse_response,
            document_path,
            output_base_dir=save_to,
        )

    (save_to / "full_detail.json").write_text(
        parse_response.to_json(),
        encoding="utf-8",
    )
    (save_to / "readable_output.md").write_text(
        getattr(parse_response, "markdown", "") or "",
        encoding="utf-8",
    )

    return parse_response


# ----------------------------- Mistral OCR Helpers -----------------------------

def save_mistralocr_outputs(
    ocr_response,
    document_path=None,
    save_to="./output",
    draw_boxes=True,
    save_images=True,
):
    """Save Mistral OCR markdown, JSON, extracted images, and layout overlay."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    (save_to / "full_detail.json").write_text(
        ocr_response.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (save_to / "readable_output.md").write_text(
        "\n\n---\n\n".join(page.markdown.strip() for page in ocr_response.pages),
        encoding="utf-8",
    )

    if save_images:
        save_mistralocr_images(ocr_response, save_to=save_to)

    if draw_boxes and document_path:
        save_mistralocr_layout_overlay(
            ocr_response,
            document_path,
            save_to=save_to,
        )

    return ocr_response


def save_mistralocr_images(ocr_response, save_to="./output"):
    """Save image assets returned by Mistral OCR."""
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for page_number, page in enumerate(ocr_response.pages, start=1):
        for image_index, image in enumerate(page.images, start=1):
            image_base64 = image.image_base64
            if not image_base64:
                continue

            image_bytes = b64decode(image_base64.split(",", 1)[-1])

            filename = (
                Path(image.id).name
                if image.id
                else f"page_{page_number}_image_{image_index}.png"
            )
            output_path = save_to / filename
            if output_path.exists():
                output_path = save_to / f"page_{page_number}_{filename}"

            output_path.write_bytes(image_bytes)
            print(f"Saved Mistral OCR image: {output_path}")
            output_paths.append(output_path)

    return output_paths


def save_mistralocr_layout_overlay(
    ocr_response,
    document_path,
    save_to="./output",
    dpi=PDF_RENDER_DPI,
):
    """Draw Mistral OCR block and image boxes on page images."""
    document_path = Path(document_path)
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    pages = ocr_response.pages
    output_paths = []

    for page in pages:
        page_number = page.index + 1
        page_image = _load_page_image(document_path, page_number, dpi=dpi)
        draw = ImageDraw.Draw(page_image)

        page_width = page.dimensions.width if page.dimensions else page_image.width
        page_height = page.dimensions.height if page.dimensions else page_image.height
        scale_x = page_image.width / (page_width or page_image.width)
        scale_y = page_image.height / (page_height or page_image.height)

        blocks = [(block, block.type) for block in (page.blocks or [])]
        images = [(image, "image") for image in page.images]

        for index, (region, label) in enumerate([*blocks, *images], start=1):
            coords = [
                region.top_left_x,
                region.top_left_y,
                region.bottom_right_x,
                region.bottom_right_y,
            ]
            if any(coord is None for coord in coords):
                continue

            x1, y1, x2, y2 = (
                int(coords[0] * scale_x),
                int(coords[1] * scale_y),
                int(coords[2] * scale_x),
                int(coords[3] * scale_y),
            )
            color = MISTRALOCR_LABEL_COLORS.get(
                str(label).lower(),
                (128, 128, 128),
            )

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            tag = f"{index}:{label}"
            label_y = max(0, y1 - 18)
            draw.rectangle([x1, label_y, x1 + len(tag) * 7, y1], fill=color)
            draw.text((x1 + 2, label_y + 2), tag, fill=(255, 255, 255))

        if len(pages) == 1:
            output_path = save_to / "layout_overlay.png"
        else:
            output_path = save_to / f"layout_overlay_page_{page_number}.png"

        page_image.save(output_path)
        print(f"Saved Mistral OCR bbox image: {output_path}")
        output_paths.append(output_path)

    return output_paths

# ----------------------------- LlamaCloud Helpers -----------------------------

def save_llamacloud_outputs(parse_response, document_path=None, save_to="./output"):
    """Save LlamaCloud markdown, text, and grounded items."""
    # Read single-page outputs.
    markdown = parse_response.markdown_full or ""
    if not markdown and parse_response.markdown:
        markdown = parse_response.markdown.pages[0].markdown

    text = parse_response.text_full or ""
    if not text and parse_response.text:
        text = parse_response.text.pages[0].text

    metadata = parse_response.metadata if parse_response.metadata else {}

    # Persist parsed content.
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)
    print(metadata)
    (save_to / "readable_output.md").write_text(markdown or "", encoding="utf-8")
    (save_to / "output.txt").write_text(text or "", encoding="utf-8")
    save_llamacloud_images(parse_response, save_to=save_to)

    # Persist grounded items sidecar.
    grounded_items = (parse_response.result_content_metadata or {}).get(
        "grounded_items"
    )
    grounded_items_url = grounded_items.presigned_url if grounded_items else None

    if grounded_items_url:
        grounded_items_path = save_to / "full_detail.jsonl"
        with urlopen(grounded_items_url) as response:
            grounded_items_path.write_bytes(response.read())

        grounded_items_json_path = save_to / "full_detail.json"
        with grounded_items_path.open(encoding="utf-8") as file:
            grounded_items = [json.loads(line) for line in file if line.strip()]
        grounded_items_json_path.write_text(
            json.dumps(grounded_items, indent=2),
            encoding="utf-8",
        )

        if document_path:
            save_llamacloud_layout_overlay(
                document_path,
                grounded_items_path,
                save_to=save_to,
            )

    return parse_response


def save_llamacloud_images(parse_response, save_to="./output"):
    """Download LlamaCloud image assets from presigned URLs."""
    images_metadata = parse_response.images_content_metadata
    if not images_metadata:
        return []

    save_to = Path(save_to)
    output_paths = []

    for image in images_metadata.images:
        image_url = image.presigned_url
        if not image_url:
            continue

        category = image.category or "uncategorized"
        filename = Path(image.filename or f"image_{image.index}.png").name
        output_path = save_to / "images" / category / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(image_url) as response:
            output_path.write_bytes(response.read())

        print(f"Saved LlamaCloud image: {output_path}")
        output_paths.append(output_path)

    return output_paths


def save_llamacloud_layout_overlay(
    document_path,
    grounded_items_path,
    save_to="./output",
    dpi=PDF_RENDER_DPI,
):
    """Draw LlamaCloud item-level region boxes on page images."""
    document_path = Path(document_path)
    grounded_items_path = Path(grounded_items_path)
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    with grounded_items_path.open(encoding="utf-8") as file:
        pages = [json.loads(line) for line in file if line.strip()]

    output_paths = []
    for page in pages:
        page_number = page.get("page_number", 1)
        page_image = _load_page_image(document_path, page_number, dpi=dpi)
        draw = ImageDraw.Draw(page_image)

        page_width = page.get("page_width") or page_image.width
        page_height = page.get("page_height") or page_image.height
        scale_x = page_image.width / page_width
        scale_y = page_image.height / page_height

        # Draw detected item regions.
        for index, item in enumerate(page.get("items", []), start=1):
            item_type = item.get("type", "item")
            color = LLAMA_ITEM_TYPE_COLORS.get(item_type, (128, 128, 128))

            for bbox in item.get("bbox", []):
                x1 = int(bbox["x"] * scale_x)
                y1 = int(bbox["y"] * scale_y)
                x2 = int((bbox["x"] + bbox["w"]) * scale_x)
                y2 = int((bbox["y"] + bbox["h"]) * scale_y)

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                label = f"{index}:{item_type}"
                label_y = max(0, y1 - 18)
                draw.rectangle([x1, label_y, x1 + len(label) * 7, y1], fill=color)
                draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))

        if len(pages) == 1:
            output_path = save_to / "layout_overlay.png"
        else:
            output_path = save_to / f"layout_overlay_page_{page_number}.png"

        page_image.save(output_path)
        print(f"Saved bbox image: {output_path}")
        output_paths.append(output_path)

    return output_paths


def _load_page_image(document_path, page_number, dpi=PDF_RENDER_DPI):
    """Load a document page as an RGB image."""
    if document_path.suffix.lower() == ".pdf":
        image_paths = pdf_to_images(document_path, dpi=dpi)
        with Image.open(image_paths[page_number - 1]) as image:
            return image.copy()

    image = Image.open(document_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
