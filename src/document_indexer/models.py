from dataclasses import dataclass, field, asdict
from hashlib import sha256
import json

def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

@dataclass
class Section:
    heading_path: list[str]
    start_char: int
    end_char: int

@dataclass
class Document:
    source: str
    title: str
    source_url: str | None
    format: str
    text: str
    content_hash: str
    lines: list[int]
    pages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    source: str
    source_url: str | None
    title: str
    sections: list[list[str]]
    ordinal: int
    text: str
    start_char: int
    end_char: int
    line_start: int
    line_end: int
    page_start: int | None
    page_end: int | None
    token_count: int
    content_hash: str

def to_dict(obj):
    return asdict(obj)
