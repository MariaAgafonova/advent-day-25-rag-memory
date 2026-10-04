from pathlib import Path
import json
import re
import unicodedata
from .models import Document, digest, canonical
from .normalize import normalize, line_map

SUPPORTED = {".md", ".txt", ".pdf"}
MAX_UPLOAD = 20 * 1024 * 1024

def load_document(path, root, metadata=None):
    path, root = Path(path), Path(root)
    meta = metadata or {}
    suffix = path.suffix.lower()
    pages, warnings = [], []
    if suffix in {".md", ".txt"}:
        try:
            text = normalize(path.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path.name}: требуется UTF-8 (или UTF-8 BOM); неверный байт {exc.start}") from exc
    elif suffix == ".pdf":
        import pymupdf
        parts, cursor = [], 0
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"{path.name}: PDF защищён паролем")
            for number, page in enumerate(pdf, 1):
                part = normalize(page.get_text("text", sort=True))
                if number > 1:
                    parts.append("\n\n")
                    cursor += 2
                pages.append({"number": number, "start_char": cursor, "end_char": cursor + len(part)})
                parts.append(part)
                cursor += len(part)
                if not part.strip():
                    warnings.append(f"Страница {number}: нет текста; возможно, требуется OCR")
        text = "".join(parts)
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")
    if not text.strip():
        warnings.append("Пустой документ")
    return Document(path.relative_to(root).as_posix(), meta.get("title", path.stem), meta.get("url"),
                    suffix[1:], text, digest(text), line_map(text), pages, warnings)

def load_corpus(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Папка корпуса не найдена: {root}")
    manifest = root.parent / "sources.json"
    metadata = {m["path"]: m for m in json.loads(manifest.read_text(encoding="utf-8"))} if manifest.exists() else {}
    documents, errors = [], []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            if not path.resolve().is_relative_to(root.resolve()):
                errors.append({"source": path.name, "error": "Ссылка за пределы корпуса"})
                continue
            try:
                documents.append(load_document(path, root, metadata.get(path.relative_to(root).as_posix())))
            except Exception as exc:
                errors.append({"source": path.relative_to(root).as_posix(), "error": str(exc)})
    return documents, errors

def corpus_hash(documents):
    return digest(canonical(sorted((d.source, d.content_hash, d.title, d.source_url) for d in documents)))

def corpus_stats(documents, errors, tokenizer=None):
    seen, duplicates = {}, []
    words = tokens = pdf_pages = text_words = 0
    unique_words = unique_pages = 0
    for doc in documents:
        count = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", doc.text))
        words += count
        pdf_pages += len(doc.pages)
        if doc.format != "pdf":
            text_words += count
        if tokenizer:
            tokens += tokenizer.count(doc.text)
        if doc.content_hash in seen:
            duplicates.append({"source": doc.source, "duplicate_of": seen[doc.content_hash]})
        else:
            seen[doc.content_hash] = doc.source
            unique_words += count
            unique_pages += sum(bool(doc.text[p["start_char"]:p["end_char"]].strip()) for p in doc.pages) if doc.pages else count / 400
    return {"files": len(documents) + len(errors), "loaded_files": len(documents), "words": words,
            "unique_words": unique_words, "tokens": tokens if tokenizer else None,
            "pdf_pages": pdf_pages, "text_equivalent_pages": text_words / 400,
            "unique_equivalent_pages": unique_pages, "volume_requirement_met": unique_pages >= 25,
            "empty_files": [d.source for d in documents if not d.text.strip()], "errors": errors,
            "duplicates": duplicates, "corpus_hash": corpus_hash(documents)}

def save_upload(root, name, content):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = unicodedata.normalize("NFKC", name.replace("\\", "/").split("/")[-1])
    name = re.sub(r'[^\w.() -]', "_", name).strip(" .")
    if not name or Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError("Поддерживаются только Markdown, TXT и PDF")
    if len(content) > MAX_UPLOAD:
        raise ValueError("Лимит одного файла — 20 МБ")
    stem, suffix = Path(name).stem[:100], Path(name).suffix.lower()
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        stem = "document_" + stem
    for i in range(10000):
        target = root / f"{stem}{f'_{i}' if i else ''}{suffix}"
        if target.resolve().parent != root:
            raise ValueError("Недопустимый путь")
        try:
            with target.open("xb") as stream:
                stream.write(content)
            return target.name
        except FileExistsError:
            continue
    raise ValueError("Слишком много файлов с одинаковым именем")
