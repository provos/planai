import io
import re

import fitz
import numpy as np
import pytesseract
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def split_into_sentences(text):
    # Simple regex-based sentence splitter
    return re.split(r"(?<=[.!?])\s+", text)


def pdf_to_text(pdf_path, use_ocr=False, ocr_language="eng"):
    """
    Extract text from a PDF file.

    Args:
    - pdf_path: str - Path to the PDF file.
    - use_ocr: bool - Whether to use OCR for scanned documents.
    - ocr_language: str - Language to use for OCR, default is English.

    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    doc = fitz.open(pdf_path)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        # Extract text using PyMuPDF
        page_text = page.get_text()

        # If text is retrieved successfully, then it's likely a text-based PDF
        if page_text.strip() or not use_ocr:
            text += page_text
        else:
            # Use OCR on images in the page if text extraction is not sufficient
            for img_index, img in enumerate(doc.get_page_images(page_number)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Load image to Pillow
                image_pil = Image.open(io.BytesIO(image_bytes))

                # Use pytesseract to extract text via OCR
                ocr_text = pytesseract.image_to_string(image_pil, lang=ocr_language)
                text += ocr_text

    doc.close()
    return text


def create_semantic_chunks(
    text, min_sentences=4, max_sentences=20, buffer_size=1, threshold_percentile=65
):
    # Split into sentences
    sentences = split_into_sentences(text)

    # Combine sentences with buffer
    combined_sentences = []
    for i in range(len(sentences)):
        combined = " ".join(
            sentences[
                max(0, i - buffer_size) : min(len(sentences), i + buffer_size + 1)
            ]
        )
        combined_sentences.append(combined)

    # Get embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sentence_embeddings = embeddings.embed_documents(combined_sentences)

    # Calculate cosine distances
    distances = []
    for i in range(len(sentence_embeddings) - 1):
        similarity = cosine_similarity(
            [sentence_embeddings[i]], [sentence_embeddings[i + 1]]
        )[0][0]
        distance = 1 - similarity
        distances.append(distance)

    # Determine breakpoints
    threshold = np.percentile(distances, threshold_percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    # Create chunks
    chunks = []
    start = 0
    for end in breakpoints:
        if end - start >= min_sentences:
            chunks.append(" ".join(sentences[start : end + 1]))
            start = end + 1
        elif chunks and len(chunks[-1].split(".")) + (end - start) <= max_sentences:
            # If the current segment is too small, try to merge with the previous chunk
            chunks[-1] += " " + " ".join(sentences[start : end + 1])
            start = end + 1

    # Add any remaining sentences
    if start < len(sentences):
        if (
            chunks
            and len(chunks[-1].split(".")) + (len(sentences) - start) <= max_sentences
        ):
            chunks[-1] += " " + " ".join(sentences[start:])
        else:
            chunks.append(" ".join(sentences[start:]))

    return chunks
