NORMALIZATION = {"version": 1, "newlines": "LF", "remove_nul": True, "preserve_indent": True}

def normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

def line_map(text):
    return [0] + [i + 1 for i, char in enumerate(text) if char == "\n"]
