from bisect import bisect_right
from .models import Chunk, digest, canonical
from .sections import extract_sections, blocks
from .tokenization import validate_settings, windows

def chunk_document(doc, strategy, tokenizer, size=300, overlap=50):
    validate_settings(size, overlap)
    if strategy not in {"fixed", "structural"}:
        raise ValueError("Неизвестная стратегия")
    sections = extract_sections(doc)
    if strategy == "fixed":
        ranges = list(windows(doc.text, tokenizer, size, overlap))
    else:
        ranges = []
        for section in sections:
            pending = None
            for a, b in blocks(doc.text, section.start_char, section.end_char):
                if tokenizer.count(doc.text[a:b]) > size:
                    if pending:
                        ranges.append(pending)
                        pending = None
                    ranges.extend(windows(doc.text[a:b], tokenizer, size, overlap, a))
                elif pending and tokenizer.count(doc.text[pending[0]:b]) <= size:
                    pending = (pending[0], b)
                else:
                    if pending:
                        ranges.append(pending)
                    pending = (a, b)
            if pending:
                ranges.append(pending)
    chunks = []
    for ordinal, (a, b) in enumerate(ranges, 1):
        text = doc.text[a:b]
        if not text.strip():
            continue
        pages = [p["number"] for p in doc.pages if p["start_char"] < b and p["end_char"] > a]
        identity = [doc.source, doc.content_hash, strategy, {"size": size, "overlap": overlap}, a, b, text]
        chunks.append(Chunk(digest(canonical(identity)), strategy, doc.source, doc.source_url, doc.title,
                            [s.heading_path for s in sections if s.start_char < b and s.end_char > a],
                            ordinal, text, a, b, bisect_right(doc.lines, a), bisect_right(doc.lines, max(a, b - 1)),
                            min(pages) if pages else None, max(pages) if pages else None,
                            tokenizer.count(text), digest(text)))
    return chunks
