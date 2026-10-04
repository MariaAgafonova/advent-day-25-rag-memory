import math

def similarity_filter(chunks, threshold):
    if threshold is None or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Задайте конечный cosine-порог от -1 до 1")
    return [chunk for chunk in chunks if chunk.retrieval_score >= threshold]
