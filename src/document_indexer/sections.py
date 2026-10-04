import re
from .models import Section

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*#*\s*$")

def fenced_lines(text):
    fence = None
    cursor = 0
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        inside = fence is not None
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = run
                inside = True
            elif run[0] == fence[0] and len(run) >= len(fence) and not tail.strip():
                fence = None
        yield cursor, line, inside
        cursor += len(line)

def extract_sections(doc):
    if doc.format == "pdf":
        return [Section([f"Страница {p['number']}"], p["start_char"],
                        doc.pages[i + 1]["start_char"] if i + 1 < len(doc.pages) else len(doc.text))
                for i, p in enumerate(doc.pages)]
    if doc.format != "md":
        return [Section([doc.title], 0, len(doc.text))]
    sections, hierarchy = [], []
    start, heading = 0, ["Introduction"]
    for offset, line, fenced in fenced_lines(doc.text):
        match = HEADING.match(line) if not fenced else None
        if match:
            if offset > start:
                sections.append(Section(heading, start, offset))
            level, title = len(match[1]), match[2]
            hierarchy = [(lev, name) for lev, name in hierarchy if lev < level] + [(level, title)]
            start, heading = offset, [name for _, name in hierarchy]
    if start < len(doc.text):
        sections.append(Section(heading, start, len(doc.text)))
    return sections

def blocks(text, start, end):
    result, block_start, in_code = [], start, False
    for local, line, fenced in fenced_lines(text[start:end]):
        pos = start + local
        if fenced and not in_code and pos > block_start:
            result.append((block_start, pos))
            block_start = pos
        if not fenced and in_code and pos > block_start:
            result.append((block_start, pos))
            block_start = pos
        if not fenced and not line.strip():
            result.append((block_start, pos + len(line)))
            block_start = pos + len(line)
        in_code = fenced
    if block_start < end:
        result.append((block_start, end))
    return [(a, b) for a, b in result if text[a:b].strip()]
