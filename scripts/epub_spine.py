"""EPUB -> Markdown with "## Page N" before each spine file, so it can be cut by chapter.

any2mdhub calls this when the uploader asks for a chapter split: anydoc keeps an EPUB's equations
but marks no spine boundaries, the dashboard's convert_epub does.
usage:  python -X utf8 scripts/epub_spine.py IN.epub OUT.md     (prints the spine as a JSON list)
"""
import json
import sys
from pathlib import Path

from dashboard import convert_epub

job = {}
Path(sys.argv[2]).write_text(convert_epub(job, sys.argv[1]), encoding="utf-8")
print(json.dumps(job["spine"], ensure_ascii=False))
