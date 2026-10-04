"""Read-only inspection of saved API requests. No config, network or LLM access."""
from copy import deepcopy
from difflib import unified_diff
import json

def inspect_request(answer):
    captured = answer.get("api_request")
    if captured:
        return {"captured": True, "request": deepcopy(captured), "notice": "Сохранённое тело фактического HTTP-запроса. Заголовок Authorization не сохраняется."}
    settings = answer.get("generation_settings", {})
    body = {"model": answer.get("model"), "messages": deepcopy(answer.get("messages", []))}
    if "temperature" in settings:
        body["temperature"] = settings["temperature"]
    token_parameter = settings.get("output_token_parameter")
    if token_parameter in {"max_tokens", "max_completion_tokens"} and "max_output_tokens" in settings:
        body[token_parameter] = settings["max_output_tokens"]
    if settings.get("thinking") is not None:
        body["thinking"] = {"type": settings["thinking"]}
    return {"captured": False, "request": {"body": body},
            "notice": "Снимок HTTP-запроса отсутствует: показаны только сохранённые сообщения и известные параметры. URL и недостающие поля не восстанавливаются из текущих настроек."}

def request_json(answer):
    return json.dumps(inspect_request(answer)["request"], ensure_ascii=False, indent=2)

def readable_request(answer):
    request = inspect_request(answer)["request"]
    body = request["body"]
    metadata = {k: v for k, v in request.items() if k != "body"}
    lines = [json.dumps({**metadata, "parameters": {k: v for k, v in body.items() if k != "messages"}}, ensure_ascii=False, indent=2)]
    for position, message in enumerate(body.get("messages", []), 1):
        lines.extend([f"\n--- Сообщение {position}: {message.get('role', 'unknown')} ---", str(message.get("content", ""))])
    return "\n".join(lines)

def request_diff(without_rag, with_rag):
    return "\n".join(unified_diff(readable_request(without_rag).splitlines(), readable_request(with_rag).splitlines(),
                                  fromfile="Без RAG", tofile="С RAG", lineterm=""))

def parameter_comparison(without_rag, with_rag):
    left, right = (inspect_request(a)["request"]["body"] for a in (without_rag, with_rag))
    keys = sorted((left.keys() | right.keys()) - {"messages"})
    rows = []
    for key in keys:
        a = json.dumps(left[key], ensure_ascii=False) if key in left else "Не сохранено"
        b = json.dumps(right[key], ensure_ascii=False) if key in right else "Не сохранено"
        rows.append({"Параметр": key, "Без RAG": a, "С RAG": b, "Совпадает": key in left and key in right and left[key] == right[key]})
    return rows
