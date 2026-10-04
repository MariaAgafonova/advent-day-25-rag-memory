"""One-time removal of site UI labels from the six files downloaded in this run."""
from pathlib import Path
import json
from document_indexer.models import digest

root = Path(__file__).resolve().parents[1]
manifest_path = root / "data/sources.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for item in manifest:
    path = root / "data/raw" / item["path"]
    title = item["title"].split(" Stay organized")[0].split(" Part of Android")[0]
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    lines[0] = "# " + title + "\n"
    if len(lines) > 2 and lines[2].startswith("# " + title):
        del lines[2:4]
    text = "".join(lines)
    path.write_text(text, encoding="utf-8")
    item["title"] = title
    item["content_hash"] = digest(text)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
