import os
import re
import json
import nltk
import time
import uuid
import base64
from io import BytesIO
from pathlib import Path
from transformers import AutoTokenizer
from langchain_core.documents import Document
from PIL import Image, UnidentifiedImageError
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from concurrent.futures import as_completed, ProcessPoolExecutor
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption

from llm_utils import filter_with_llm, classify_text_with_llm, summarize_table
from vision_utils import generate_image_summary


nltk.download("punkt")
nltk.download('punkt_tab')


IMAGE_RESOLUTION_SCALE = 1.0
excluded_labels = {
    'page_header', 'page_footer', 'caption', 'reference'
}

def process_document(res, out_path, gen_model, gen_endpoint, vlm_model, vlm_endpoint, vlm_hosting_type, start_time, timings):
    doc_json = res.document.export_to_dict()
    stem = res.input.file.stem

    # --- Text Extraction ---
    t0 = time.time()
    filtered_blocks, image_captions, table_captions = [], [], []
    for block in doc_json.get('texts', []):
        block_type = block.get('label', '')
        if block_type not in excluded_labels:
            filtered_blocks.append(block)
        if block_type == 'caption':
            block_parent = block.get('parent', {}).get('$ref', '')
            if 'tables' in block_parent:
                table_captions.append(block)
            elif 'pictures' in block_parent:
                image_captions.append(block)
    timings['extract_text_blocks'] = time.time() - t0

    if len(filtered_blocks):
        t0 = time.time()
        filtered_text_dicts = filter_with_llm(filtered_blocks, gen_model, gen_endpoint)
        (Path(out_path) / f"{stem}_clean_text.json").write_text(json.dumps(filtered_text_dicts, indent=2), encoding="utf-8")
        timings['llm_filter_text'] = time.time() - t0
    else:
        (Path(out_path) / f"{stem}_clean_text.json").write_text(json.dumps(filtered_blocks, indent=2), encoding="utf-8")

    # --- Image Extraction ---
    if len(doc_json.get('pictures', [])):
        t0 = time.time()
        image_dict, image_uris, ordered_image_captions = [], [], []
        for image_idx, block in enumerate(doc_json.get('pictures', [])):
            caption = ''
            for child in block.get('children', []):
                child_id = child['$ref']
                for caption_idx, child_block in enumerate(image_captions):
                    if child_block.get('self_ref', '') == child_id:
                        caption += f'{child_block["text"]} '
                        image_captions.pop(caption_idx)
                        break
            uri = block.get('image', {}).get('uri', '')
            image_uris.append(uri)
            ordered_image_captions.append(caption)
        timings['extract_images'] = time.time() - t0

        t0 = time.time()
        image_summaries = generate_image_summary(list(zip(image_uris, ordered_image_captions)), vlm_model, vlm_endpoint, vlm_hosting_type)
        timings['generate_image_summaries'] = time.time() - t0

        t0 = time.time()
        for idx, (uri, summary, caption) in enumerate(zip(image_uris, image_summaries, ordered_image_captions)):
            image_dict.append({idx: {'image': uri, 'caption': caption, 'summary': summary}})
        decisions = classify_text_with_llm(image_summaries, gen_model, gen_endpoint)
        filtered_image_dicts = [image_dict[idx] for idx, keep in enumerate(decisions) if keep]
        (Path(out_path) / f"{stem}_images.json").write_text(json.dumps(filtered_image_dicts, indent=2), encoding="utf-8")
        timings['filter_image_summaries'] = time.time() - t0
    else:
        (Path(out_path) / f"{stem}_images.json").write_text(json.dumps([], indent=2), encoding="utf-8")

    # --- Table Extraction ---
    if len(res.document.tables):
        t0 = time.time()
        table_htmls_dict = {}
        table_captions_dict = {i: None for i in range(len(res.document.tables))}
        for table_ix, table in enumerate(res.document.tables):
            table_htmls_dict[table_ix] = table.export_to_html(doc=res.document)
            for caption_idx, block in enumerate(table_captions):
                if block.get('parent')['$ref'] == f'#/tables/{table_ix}':
                    table_captions_dict[table_ix] = block.get('text', '')
                    table_captions.pop(caption_idx)
                    break
        table_htmls = [table_htmls_dict[key] for key in sorted(table_htmls_dict)]
        table_captions_list = [table_captions_dict[key] for key in sorted(table_captions_dict)]
        timings['extract_tables'] = time.time() - t0

        t0 = time.time()
        table_summaries = summarize_table(table_htmls, table_captions_list, gen_model, gen_endpoint)
        timings['summarize_tables'] = time.time() - t0

        t0 = time.time()
        decisions = classify_text_with_llm(table_summaries, gen_model, gen_endpoint)
        filtered_table_dicts = {
            idx: {
                'html': html,
                'caption': caption,
                'summary': summary
            }
            for idx, (keep, html, caption, summary) in enumerate(zip(decisions, table_htmls, table_captions_list, table_summaries)) if keep
        }
        (Path(out_path) / f"{stem}_tables.json").write_text(json.dumps(filtered_table_dicts, indent=2), encoding="utf-8")
        timings['filter_tables'] = time.time() - t0
    else:
        (Path(out_path) / f"{stem}_tables.json").write_text(json.dumps([], indent=2), encoding="utf-8")

    total_time = time.time() - start_time
    print(f"\n[Timing for {stem}] Total: {total_time:.2f}s")
    for k, v in timings.items():
        print(f"  {k:<30}: {v:.2f}s")

def convert_and_process(path, doc_converter, out_path, llm_model, llm_endpoint, vlm_model, vlm_endpoint, vlm_hosting_type):
    try:
        start_time = time.time()
        timings = {}
        t0 = time.time()
        res = doc_converter.convert(path)
        timings['conversion_time'] = time.time() - t0
        process_document(res, out_path, llm_model, llm_endpoint, vlm_model, vlm_endpoint, vlm_hosting_type, start_time, timings)
    except Exception as e:
        print(f"Error converting or processing {path}: {e}")


def extract_document_data(input_paths, out_path, llm_model, llm_endpoint, vlm_model, vlm_endpoint, vlm_hosting_type, force=False):
    # Accelerator & pipeline options
    accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CUDA)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_picture_images = True

    # Skip files that already exist
    filtered_input_paths = [
        path for path in input_paths if force or not (
            (Path(out_path) / f"{Path(path).stem}_clean_text.json").exists() and
            (Path(out_path) / f"{Path(path).stem}_images.json").exists() and
            (Path(out_path) / f"{Path(path).stem}_tables.json").exists()
        )
    ]
    print(f"Processing {len(filtered_input_paths)} files...")

    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX, InputFormat.HTML,
            InputFormat.PPTX, InputFormat.ASCIIDOC, InputFormat.CSV, InputFormat.MD,
        ],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    if filtered_input_paths:
        with ProcessPoolExecutor(max_workers=max(1, min(4, len(filtered_input_paths)))) as executor:
            futures = [
                executor.submit(convert_and_process, path, doc_converter, out_path, llm_model, llm_endpoint, vlm_model, vlm_endpoint, vlm_hosting_type)
                for path in input_paths
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Unhandled exception: {e}")
    else:
        print("No files to process.")


def get_header_level(text):
    chapter_match = re.match(r"^(\d+)\s+(.*)", text)
    section_match = re.match(r"^(\d+\.\d+)\s+(.*)", text)
    subsection_match = re.match(r"^(\d+\.\d+\.\d+)\s+(.*)", text)
    if subsection_match:
        return 3, subsection_match.group(0).strip()
    elif section_match:
        return 2, section_match.group(0).strip()
    elif chapter_match:
        return 1, chapter_match.group(0).strip()
    return 0, None


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def split_text_into_token_chunks(text, tokenizer, max_tokens=512, overlap=50):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        token_len = count_tokens(sentence, tokenizer)

        if current_token_count + token_len > max_tokens:
            # save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            # overlap logic (optional)
            if overlap > 0 and len(current_chunk) > 0:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_token_count = count_tokens(sentence, tokenizer)
            else:
                current_chunk = []
                current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += token_len

    # flush last
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def flush_chunk(current_chunk, chunks, tokenizer, max_tokens):
    content = current_chunk["content"].strip()
    if not content:
        return

    # Split content into token chunks
    token_chunks = split_text_into_token_chunks(content, tokenizer, max_tokens=max_tokens)

    for i, part in enumerate(token_chunks):
        chunk = {
            "chapter_title": current_chunk["chapter_title"],
            "section_title": current_chunk["section_title"],
            "subsection_title": current_chunk["subsection_title"],
            "content": part,
            "page_range": sorted(set(current_chunk["page_range"])),
            "source_nodes": current_chunk["source_nodes"].copy()
        }
        if len(token_chunks) > 1:
            chunk["part_id"] = i + 1
        chunks.append(chunk)

    # Reset current_chunk after flushing
    current_chunk["content"] = ""
    current_chunk["page_range"] = []
    current_chunk["source_nodes"] = []


def process_single_file(input_path, output_path, tokenizer, max_tokens=512):
    print(f"Processing {input_path} -> {output_path}")
    
    if not Path(output_path).exists():
        with open(input_path, "r") as f:
            data = json.load(f)

        chunks = []
        current_chunk = {
            "chapter_title": None,
            "section_title": None,
            "subsection_title": None,
            "content": "",
            "page_range": [],
            "source_nodes": []
        }

        current_chapter = None
        current_section = None
        current_subsection = None

        for idx, block in enumerate(data):
            label = block.get("label")
            text = block.get("text", "").strip()
            try:
                page_no = block.get("prov", {})[0].get("page_no")
            except:
                page_no = 0
            ref = f"#texts/{idx}"

            if label == "section_header":
                level, full_title = get_header_level(text)
                if level == 1:
                    current_chapter = full_title
                    current_section = None
                    current_subsection = None
                elif level == 2:
                    current_section = full_title
                    current_subsection = None
                elif level == 3:
                    current_subsection = full_title

                # Flush current chunk and update
                flush_chunk(current_chunk, chunks, tokenizer, max_tokens)
                current_chunk["chapter_title"] = current_chapter
                current_chunk["section_title"] = current_section
                current_chunk["subsection_title"] = current_subsection

            elif label in {"text", "list_item"}:
                if current_chunk["chapter_title"] is None:
                    current_chunk["chapter_title"] = current_chapter
                if current_chunk["section_title"] is None:
                    current_chunk["section_title"] = current_section
                if current_chunk["subsection_title"] is None:
                    current_chunk["subsection_title"] = current_subsection

                current_chunk["content"] += text + " "
                if page_no is not None:
                    current_chunk["page_range"].append(page_no)
                current_chunk["source_nodes"].append(ref)

        # Flush any remaining content
        flush_chunk(current_chunk, chunks, tokenizer, max_tokens)

        # Save the processed chunks to the output file
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)

        print(f"✅ {len(chunks)} RAG chunks saved to {output_path}")
    else:
        print(f"{output_path} already exists.")

def hierarchical_chunk_with_token_split(input_paths, output_paths, max_tokens=512):
    if len(input_paths) != len(output_paths):
        raise ValueError("`input_paths` and `output_paths` must have the same length")

    tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-embedding-278m-multilingual')

    # Process each input-output file pair in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for input_path, output_path in zip(input_paths, output_paths):
            print(f"Submitting task for: {input_path} -> {output_path}")
            futures.append(executor.submit(process_single_file, input_path, output_path, tokenizer, max_tokens))

        # Wait for all futures to finish and handle exceptions
        for future in futures:
            try:
                future.result()  # Capture exceptions if any
            except Exception as e:
                print(f"Error occurred: {e}")


def create_chunk_documents(in_txt_f, in_img_f, in_tab_f, orig_fn, include_meta_info_in_main_text, collection_name):

    os.makedirs(f'images_{collection_name}', exist_ok=True)

    with open(in_txt_f, "r") as f:
        txt_data = json.load(f)

    with open(in_img_f, "r") as f:
        img_data = json.load(f)

    with open(in_tab_f, "r") as f:
        tab_data = json.load(f)

    txt_docs = []
    if len(txt_data):
        for txt_id, block in enumerate(txt_data):
            meta_info = ''
            if block.get('chapter_title'):
                meta_info += f"Chapter: {block.get('chapter_title')} "
            if block.get('section_title'):
                meta_info += f"Section: {block.get('section_title')} "
            if block.get('subsection_title'):
                meta_info += f"Subsection: {block.get('subsection_title')} "
            txt_docs.append(Document(
                page_content=f'{meta_info}\n{block.get("content")}' if include_meta_info_in_main_text else block.get("content"),
                metadata={"filename": orig_fn, "type": "text", "source": meta_info, "chunk_id": txt_id}
            ))

    img_docs = []
    if len(img_data):
        for img_id, block in enumerate(img_data):
            block = list(block.values())[0]
            img_path = f"images_{collection_name}/{uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_DNS, collection_name), f'{orig_fn.strip().lower()}_img_{str(img_id).strip()}').hex}.png"

            uri = block.get('image')
            # Split off the header if present
            if ',' in uri:
                _, b64_data = uri.split(',', 1)
            else:
                b64_data = uri
            try:
                # Decode base64 string safely
                img_data = base64.b64decode(b64_data)
                # Load the image using PIL
                image = Image.open(BytesIO(img_data))
                image.load()  # Ensure the image is fully loaded
                image.save(img_path)
                img_docs.append(Document(
                page_content=block.get('summary'),
                    metadata={"filename": orig_fn, "type": "image", "source": img_path, "chunk_id": img_id}
                ))
            except base64.binascii.Error:
                print("❌ Error: The base64 data is invalid.")
            except UnidentifiedImageError:
                print("❌ Error: Cannot identify image file. The data might not be a valid image.")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

    
    tab_docs = []
    if len(tab_data):
        tab_data = list(tab_data.values())
        for tab_id, block in enumerate(tab_data):
            tab_docs.append(Document(
                page_content=block.get('summary'),
                metadata={"filename": orig_fn, "type": "table", "source": block.get('html'), "chunk_id": tab_id}
            ))

    combined_docs = txt_docs + img_docs + tab_docs

    stats = f'{len(txt_docs)} Text Chunks, {len(img_docs)} Images, and {len(tab_docs)} Tables.'

    return combined_docs, stats
