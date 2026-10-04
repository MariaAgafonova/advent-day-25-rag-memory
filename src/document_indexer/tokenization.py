class Tokenization:
    def __init__(self, tokenizer):
        if not tokenizer.is_fast:
            raise ValueError("Нужен fast tokenizer с offset_mapping")
        self.tokenizer = tokenizer

    def offsets(self, text):
        result = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return [(a, b) for a, b in result["offset_mapping"] if b > a]

    def count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

def validate_settings(size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Размер чанка > 0; 0 ≤ overlap < размера чанка")

def windows(text, tokenizer, size, overlap, base=0):
    validate_settings(size, overlap)
    offsets = tokenizer.offsets(text)
    cursor = 0
    while cursor < len(offsets):
        stop = min(cursor + size, len(offsets))
        a = 0 if cursor == 0 else offsets[cursor][0]
        b = len(text) if stop == len(offsets) else offsets[stop][0]
        yield base + a, base + b
        if stop == len(offsets):
            break
        cursor += size - overlap
