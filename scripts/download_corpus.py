"""Explicit network operation, never called during indexing. Official Android docs."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

PAGES = [
    ("lifecycle", "https://developer.android.com/guide/components/activities/activity-lifecycle"),
    ("fragment-lifecycle", "https://developer.android.com/guide/fragments/lifecycle"),
    ("viewmodel", "https://developer.android.com/topic/libraries/architecture/viewmodel"),
    ("coroutines", "https://developer.android.com/kotlin/coroutines"),
    ("workmanager", "https://developer.android.com/develop/background-work/background-tasks/persistent"),
    ("deep-links", "https://developer.android.com/training/app-links/deep-linking"),
    ("architecture", "https://developer.android.com/topic/architecture"),
]

def main():
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    for slug, url in PAGES:
        path = raw / (slug + ".md")
        if path.exists():
            print(f"Сохранён существующий файл: {path.name}")
            continue
        try:
            response = requests.get(url, params={"hl": "en"}, timeout=90)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.select_one(".devsite-article-body")
            if article is None:
                raise ValueError("Основной текст страницы не найден")
            title = soup.select_one("h1").get_text(" ", strip=True).split(" Stay organized")[0].split(" Part of Android")[0]
            for element in article.select("script,style,nav,devsite-feedback,.devsite-code-buttons"):
                element.decompose()
            for heading in article.select("h1"):
                heading.decompose()
            text = "# " + title + "\n\n" + markdownify(str(article), heading_style="ATX", code_language="kotlin")
            text = re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"
            if len(text.split()) < 150:
                raise ValueError("Подозрительно короткий ответ сервера")
            path.write_text(text, encoding="utf-8")
            manifest.append({"path": path.name, "title": title, "url": url,
                             "retrieved_at": datetime.now(timezone.utc).isoformat(), "language": "en",
                             "license": "CC BY 4.0 (text), Apache 2.0 (code)", "publisher": "Google / Android Developers"})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{path.name}: {len(text.split())} whitespace-separated words", flush=True)
        except Exception as exc:
            print(f"Не удалось получить {url}: {exc}", flush=True)

if __name__ == "__main__":
    main()
