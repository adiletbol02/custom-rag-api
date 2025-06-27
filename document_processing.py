# document_processing.py

import os
import logging
import re
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_core.documents import Document
import pdfplumber
from PyPDF2 import PdfReader

# Suppress pdfminer and pdfplumber debug logs
logging.getLogger("pdfminer").setLevel(logging.INFO)
logging.getLogger("pdfplumber").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def clean_pdf_text(file_path):
    """
    Extract and clean text from a PDF file using pdfplumber for better OCR handling.
    Returns cleaned text and page-level metadata in Chroma-compatible format.
    """
    try:
        text = ""
        page_numbers = []
        total_char_count = 0
        metadata = {"source": os.path.basename(file_path)}
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text from the page
                page_text = page.extract_text() or ""

                # Skip pages with no meaningful content
                if not page_text.strip():
                    logger.debug(f"Page {page_num} in {file_path} is empty or unreadable")
                    continue

                # Clean OCR artifacts
                # Remove repeated lines (e.g., "KOMUHA" repeated multiple times)
                lines = page_text.split('\n')
                cleaned_lines = []
                seen_lines = set()
                for line in lines:
                    line = line.strip()
                    if line and line not in seen_lines:
                        cleaned_lines.append(line)
                        seen_lines.add(line)

                # Remove special characters and normalize whitespace
                cleaned_text = ' '.join(cleaned_lines)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize multiple spaces
                cleaned_text = re.sub(r'[^\w\s.,;:!?()-]', '', cleaned_text)  # Remove special chars
                cleaned_text = re.sub(r'\d+\.\d+-\d+', '0.0', cleaned_text)  # Handle invalid floats

                # Skip pages with minimal content (e.g., less than 100 chars after cleaning)
                if len(cleaned_text.strip()) < 100:
                    logger.debug(f"Page {page_num} in {file_path} has minimal content after cleaning")
                    continue

                text += f"\n[Page {page_num}]\n{cleaned_text}\n"
                page_numbers.append(str(page_num))
                total_char_count += len(cleaned_text)

        # Skip documents with insufficient total content
        if len(text.strip()) < 100:
            logger.warning(f"No meaningful content extracted from {file_path}")
            return "", metadata

        # Add simplified page metadata
        metadata["page_numbers"] = ",".join(page_numbers)  # e.g., "2,3,4,..."
        metadata["total_char_count"] = total_char_count  # Total characters across pages

        logger.info(f"Extracted and cleaned text from {file_path}: {len(text)} characters")
        return text, metadata
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        return "", {"source": os.path.basename(file_path)}

def load_document(file_path): # Changed from load_documents (plural) to load_document (singular)
    """
    Load a single document, handling multiple file types.
    For PDFs, use clean_pdf_text to extract and clean content.
    """
    logger.info(f"Checking file: {os.path.abspath(file_path)}")
    documents = []
    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,  # Fallback, overridden by clean_pdf_text
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader
    }

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in loaders:
        try:
            if file_ext == ".pdf":
                # Unpack the tuple returned by clean_pdf_text
                text, metadata = clean_pdf_text(file_path)
                # Verify types
                if not isinstance(text, str):
                    logger.error(f"Text for {file_path} is not a string: {type(text)}")
                if not isinstance(metadata, dict):
                    logger.error(f"Metadata for {file_path} is not a dict: {type(metadata)}")
                if text.strip():  # Only add if there's meaningful content
                    documents.append(Document(page_content=text, metadata=metadata))
                    logger.info(f"Loaded {os.path.basename(file_path)}: 1 document with {len(metadata.get('page_numbers', '').split(','))} pages")
                else:
                    logger.warning(f"No content loaded for {os.path.basename(file_path)}")
            else:
                loader = loaders[file_ext](file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info(f"Loaded {os.path.basename(file_path)}: {len(loaded_docs)} documents")
        except Exception as e:
            logger.error(f"Error loading {os.path.basename(file_path)}: {e}")

    return documents