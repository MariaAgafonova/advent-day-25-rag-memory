import re

def validate_citations(answer, items):
    known = {item.citation_id for item in items}
    found = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)))
    return [ref for ref in found if ref in known], [ref for ref in found if ref not in known]
