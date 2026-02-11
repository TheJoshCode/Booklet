import re
import logging
from typing import Iterator
from pypdf import PdfReader

logger = logging.getLogger("booklet.parser")

def load_text(file_path: str) -> str:
    logger.debug(f"Loading text from: {file_path}")
    if file_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages_text)
            logger.info(f"PDF loaded — {len(text):,} chars from {len(pages_text)} pages")
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}", exc_info=True)
            return ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logger.info(f"TXT loaded — {len(text):,} chars")
        return text
    except Exception as e:
        logger.error(f"Error reading TXT {file_path}: {e}")
        return ""

def normalize_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    logger.debug(f"Text normalized — length: {len(text)}")
    return text

def chunk_text(text: str, chunk_size: int) -> Iterator[str]:
    text = normalize_text(text)
    if not text:
        logger.warning("No text to chunk")
        return

    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current_chunk:
                yield " ".join(current_chunk)
            current_chunk, current_length = [], 0
            words = sentence.split()
            for word in words:
                if current_length + len(word) + 1 > chunk_size:
                    yield " ".join(current_chunk)
                    current_chunk, current_length = [word], len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            continue

        if current_length + len(sentence) + 1 > chunk_size:
            if current_chunk:
                yield " ".join(current_chunk)
            current_chunk, current_length = [sentence], len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1

    if current_chunk:
        yield " ".join(current_chunk)