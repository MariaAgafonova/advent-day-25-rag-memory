from bisect import bisect_right
from dataclasses import dataclass, field
from .rag_models import ContextItem
from .prompts import messages_with_rag

class TokenCounter:
    def __init__(self, model):
        self.encoding = None
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model(model)
        except (KeyError, ImportError, OSError):
            pass
        self.method = "model tokenizer; message overhead conservative estimate" if self.encoding else "conservative estimate: UTF-8 bytes (not exact LLM tokens)"

    def count(self, text):
        return len(self.encoding.encode(text, disallowed_special=())) if self.encoding else len(text.encode("utf-8"))

    def messages(self, messages):
        # Provider chat framing is not assumed exact even with known text tokenizer.
        return 128 + sum(self.count(m["content"]) + 16 for m in messages)

def format_context(items):
    parts = []
    for item in items:
        location = f"pages {item.page_start}-{item.page_end}" if item.page_start else f"lines {item.line_start}-{item.line_end}"
        parts.append(f"[{item.citation_id}]\nTitle: {item.title}\nSource: {item.source}\nSection: {'; '.join(' / '.join(s) for s in item.section)}\nLocation: {location}\nText:\n{item.included_text}")
    return "\n\n".join(parts)

def subtract_ranges(start, end, covered):
    spans = [(start, end)]
    for a, b in covered:
        spans = [(x, y) for left, right in spans for x, y in [(left, min(right, a)), (max(left, b), right)] if x < y]
    return spans

@dataclass
class ContextBundle:
    items: list[ContextItem]
    text: str
    token_count: int
    token_count_method: str
    exclusions: dict = field(default_factory=dict)

class ContextBuilder:
    def __init__(self, counter):
        self.counter = counter

    def build(self, retrieved, question, settings, documents, message_factory=messages_with_rag, allow_truncation=True):
        settings.validate()
        if self.counter.messages(message_factory(question, "")) + settings.max_output_tokens > settings.context_window_tokens:
            raise ValueError("Вопрос и инструкции не помещаются в окно модели с резервом ответа")
        items, covered, seen = [], {}, set()
        exclusions = {}
        for chunk in retrieved:
            if chunk.text in seen:
                exclusions[chunk.chunk_id] = "duplicate_text"
                continue
            seen.add(chunk.text)
            doc = documents[chunk.source]
            key = (chunk.source, chunk.document_hash)
            spans = subtract_ranges(chunk.start_char, chunk.end_char, covered.get(key, []))
            if not spans:
                exclusions[chunk.chunk_id] = "covered_range"
            before = len(items)
            for a, b in spans:
                def candidate(end):
                    pages = [p["number"] for p in doc.pages if p["start_char"] < end and p["end_char"] > a]
                    text = doc.text[a:end]
                    return ContextItem(f"S{len(items)+1}", chunk.chunk_id, text, chunk.source, chunk.title,
                        chunk.section, bisect_right(doc.lines, a), bisect_right(doc.lines, max(a, end-1)),
                        min(pages) if pages else None, max(pages) if pages else None, a, end, chunk.document_hash,
                        self.counter.count(text), self.counter.method, (a, end) != (chunk.start_char, chunk.end_char), chunk.source_url)
                def fits(item):
                    context = format_context([*items, item])
                    return (self.counter.count(context) <= settings.max_context_tokens and
                            self.counter.messages(message_factory(question, context)) + settings.max_output_tokens <= settings.context_window_tokens)
                item = candidate(b)
                if not fits(item):
                    if not allow_truncation:
                        continue
                    low, high = a, b
                    while low < high:
                        mid = (low + high + 1) // 2
                        if fits(candidate(mid)):
                            low = mid
                        else:
                            high = mid - 1
                    if low == a:
                        continue
                    item = candidate(low)
                if not item.included_text.strip() or not fits(item):
                    continue
                items.append(item)
                covered.setdefault(key, []).append((item.start_char, item.end_char))
            if spans and len(items) == before:
                exclusions[chunk.chunk_id] = "context_budget"
        context = format_context(items)
        return ContextBundle(items, context, self.counter.count(context), self.counter.method, exclusions)
