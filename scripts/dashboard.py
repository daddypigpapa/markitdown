"""Local dashboard: upload a PDF/EPUB, convert it to Markdown, split it by chapter, review it.

PDFs go through the full-page OCR of ocr_pdf.py; EPUBs through markitdown. When
"split" is on, the model reads the table of contents and picks where each chapter
starts; the files are then cut at those pages. Everything lands in dashboard_data/.

setup:  pip install -e packages/markitdown-ocr openai
usage:  python -X utf8 scripts/dashboard.py [port]      (default 8123, http://127.0.0.1:8123)
        python -X utf8 scripts/dashboard.py --selftest
"""
import json
import queue
import re
import sys
import threading
import time
import traceback
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pypdfium2 as pdfium
from markitdown import StreamInfo
from markitdown.converters import EpubConverter

from check_md import check
from ocr_pdf import MODEL, OCR_EFFORT, PROMPT, LoggedOCR, PdfConverterWithOCR, load_pdf, make_client

HERE = Path(__file__).parent
DATA = HERE.parent / "dashboard_data"
JOB_ID = re.compile(r"[0-9a-f]{12}")
PAGE_HEAD = re.compile(r"^## Page (\d+)[ \t]*$", re.M)
# ponytail: the TOC is assumed to sit in the first pages; a TOC at the back of a book is missed
TOC_PAGES = 12
SPLIT_PROMPT = """Below is a book as numbered blocks. FIRST PAGES has the full text of the first blocks; \
ALL BLOCKS has how every block begins.

1. Find the table of contents (목차, 차례, Contents). A unit cover page that lists its chapters with page numbers also counts.
2. From it, decide which block each chapter starts at.

Rules:
- Split at the chapter level: the numbered chapters inside parts. Do not split at parts alone when they contain chapters, and never at sub-sections, exercises or lessons (e.g. STEP1).
- Printed page numbers differ from block numbers. Match chapters by the title at the start of a block, and use the printed numbers only as a hint for the constant offset between them.
- This may be only a slice of a book: leave out chapters that are not in these blocks.
- A part cover page right before its first chapter belongs to that chapter, even when that cover also lists the chapters.
- Aim for units a reader takes up as one. If this gives more than 30 chapters, use the level above instead (e.g. the ㄱ, ㄴ, ㄷ groups of a dictionary), and keep introductory sections together as one chapter.
- Keep titles as printed.

Return JSON only: {"toc_found": true or false, "toc_block": number or null, "chapters": [{"title": "...", "start": block number}]}
If there is no table of contents, return {"toc_found": false, "toc_block": null, "chapters": []}.
"""

jobs = {}
work = queue.Queue()


def save(job):
    (DATA / job["id"] / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")


def split_pages(md):
    """[(page number, text)]; anything before the first marker joins the first page."""
    cuts = list(PAGE_HEAD.finditer(md))
    ends = [m.start() for m in cuts[1:]] + [len(md)]
    return [(int(m[1]), md[0 if i == 0 else m.start() : ends[i]]) for i, m in enumerate(cuts)]


def head(text, size=160):
    text = PAGE_HEAD.sub("", text).replace("*[Image OCR]", "").replace("[End OCR]*", "")
    return " ".join(re.sub(r"!\[.*?\]\(.*?\)", "", text).split())[:size]


def ask_chapters(client, pages):
    first = "\n\n".join(f"[{no}]\n{text}" for no, text in pages[:TOC_PAGES])
    heads = "\n".join(f"[{no}] {head(text)}" for no, text in pages)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{SPLIT_PROMPT}\nFIRST PAGES\n{first}\n\nALL BLOCKS\n{heads}"}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


def plan_parts(pages, chapters):
    """Turn the model's chapters into [(title, first page, last page)]; raises on a bad plan."""
    starts = [c["start"] for c in chapters]
    last = pages[-1][0]
    if not starts or starts != sorted(set(starts)) or starts[0] < pages[0][0] or starts[-1] > last:
        raise ValueError(f"invalid chapter starts {starts}")
    parts = [("앞부분", pages[0][0], starts[0] - 1)] if starts[0] > pages[0][0] else []
    ends = [s - 1 for s in starts[1:]] + [last]
    return parts + [(c["title"], c["start"], end) for c, end in zip(chapters, ends)]


def file_name(index, title):
    return f"{index:02d}_{re.sub(r'[^\w-]+', '_', title).strip('_')[:60]}.md"


def write_parts(job, pages, parts):
    """Write one .md per part. Page markers mean nothing in an EPUB, so they are dropped there."""
    out = DATA / job["id"] / "md"
    out.mkdir(exist_ok=True)
    job["parts"] = []
    number = 0 if parts[0][0] == "앞부분" and len(parts) > 1 else 1
    for title, first, last in parts:
        text = "".join(t for no, t in pages if first <= no <= last)
        if job["kind"] == "epub":
            text = PAGE_HEAD.sub("", text)
        name = file_name(number, title)
        (out / name).write_text(text.strip() + "\n", encoding="utf-8")
        job["parts"].append({"file": name, "title": title, "first": first, "last": last})
        number += 1


def ocr_pdf(job, path):
    class Progress(LoggedOCR):
        def extract_text(self, image_stream, **kwargs):
            result = super().extract_text(image_stream, **kwargs)
            job["done"] += 1
            job["failed"] += bool(result.error or not result.text.strip())
            job["page_time"] = time.time()
            return result

    job["total"] = len(pdfium.PdfDocument(path))
    service = Progress(client=make_client(OCR_EFFORT), model=MODEL, default_prompt=PROMPT)
    return PdfConverterWithOCR()._ocr_full_pages(load_pdf(path, None), service)


def convert_epub(job, path):
    """markitdown joins the spine into one text; mark each spine file so it can be cut like a PDF."""
    converter = EpubConverter()
    html, spine = converter._html_converter, []

    class Marked:
        def convert(self, stream, *args, **kwargs):
            spine.append(stream.name)
            result = html.convert(stream, *args, **kwargs)
            result.markdown = f"## Page {len(spine)}\n\n{result.markdown}"
            return result

    converter._html_converter = Marked()
    with open(path, "rb") as f:
        result = converter.convert(f, StreamInfo(extension=".epub"))
    job["spine"] = spine
    return result.markdown


def run(job):
    folder = DATA / job["id"]
    source = folder / job["source"]
    job["status"], job["started"] = "running", time.time()
    save(job)
    try:
        md = ocr_pdf(job, source) if job["kind"] == "pdf" else convert_epub(job, source)
        (folder / "full.md").write_text(md, encoding="utf-8")  # OCR is slow and paid: keep it before splitting
        if job["kind"] == "pdf":
            job["flags"] = check(md)
        pages = split_pages(md)
        if not pages:
            raise ValueError("변환 결과가 비어 있습니다.")
        parts = [(Path(job["name"]).stem, pages[0][0], pages[-1][0])]
        if job["split"]:
            job["status"] = "splitting"
            try:
                plan = ask_chapters(make_client(), pages)
                if plan.get("toc_found") and plan.get("chapters"):
                    parts = plan_parts(pages, plan["chapters"])
                    job["toc_block"] = plan.get("toc_block")
                else:
                    job["note"] = "목차를 찾지 못해 파일 1개로 저장했습니다."
            except Exception as e:
                job["note"] = f"분할에 실패해 파일 1개로 저장했습니다: {e}"
        write_parts(job, pages, parts)
        job["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)
    job["finished"] = time.time()
    save(job)


def worker():
    while True:
        run(work.get())


def load_jobs():
    DATA.mkdir(exist_ok=True)
    for path in DATA.glob("*/job.json"):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] not in ("done", "error"):
            job["status"], job["error"] = "error", "서버가 재시작되어 중단됐습니다. 다시 업로드하세요."
            save(job)
        jobs[job["id"]] = job


class Handler(BaseHTTPRequestHandler):
    def send(self, body, content_type, status=200, extra=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send(json.dumps(value, ensure_ascii=False).encode(), "application/json", status)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self.send((HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/jobs":
            return self.send_json(sorted(map(dict, list(jobs.values())), key=lambda j: j["created"], reverse=True))  # copies: the worker keeps adding keys
        m = re.fullmatch(r"/files/([0-9a-f]{12})/(source|md|epub)(?:/(.+))?", path)
        job = jobs.get(m[1]) if m else None
        if not job:
            return self.send_json({"error": "not found"}, 404)
        folder = DATA / job["id"]
        if m[2] == "source":
            return self.send((folder / job["source"]).read_bytes(), "application/pdf" if job["kind"] == "pdf" else "application/epub+zip")
        if m[2] == "md":
            names = [p["file"] for p in job.get("parts", [])]
            if m[3] not in names:  # only names we wrote, so no path can escape the folder
                return self.send_json({"error": "not found"}, 404)
            return self.send((folder / "md" / m[3]).read_bytes(), "text/markdown; charset=utf-8")
        return self.send_epub_member(folder / job["source"], m[3])

    def send_epub_member(self, source, member):
        """Serve a file of the book as is; the CSP keeps the book's scripts from running."""
        types = {".html": "text/html", ".xhtml": "application/xhtml+xml", ".htm": "text/html", ".css": "text/css",
                 ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml"}
        with zipfile.ZipFile(source) as z:
            if member not in z.namelist():
                return self.send_json({"error": "not found"}, 404)
            body = z.read(member)
        kind = types.get(Path(member).suffix.lower(), "application/octet-stream")
        self.send(body, kind, extra=[("Content-Security-Policy", "script-src 'none'")])

    def do_POST(self):
        # X-File-Name is a custom header: a browser preflights it, and OPTIONS is not answered,
        # so other sites cannot start a paid conversion through this local server.
        if urlsplit(self.path).path != "/api/upload" or "X-File-Name" not in self.headers:
            return self.send_json({"error": "bad request"}, 400)
        name = Path(unquote(self.headers["X-File-Name"]).replace("\\", "/")).name
        kind = Path(name).suffix.lower().lstrip(".")
        length = int(self.headers.get("Content-Length", 0))
        if kind not in ("pdf", "epub") or not length:
            self.rfile.read(length)
            return self.send_json({"error": "PDF 또는 EPUB 파일만 올릴 수 있습니다."}, 400)
        job = {"id": uuid.uuid4().hex[:12], "name": name, "kind": kind, "source": f"source.{kind}", "created": time.time(),
               "split": self.headers.get("X-Split") == "1", "status": "queued", "done": 0, "failed": 0, "total": 0}
        folder = DATA / job["id"]
        folder.mkdir()
        with open(folder / job["source"], "wb") as f:
            while length:
                chunk = self.rfile.read(min(length, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                length -= len(chunk)
        save(job)
        jobs[job["id"]] = job
        work.put(job)
        self.send_json(job, 201)

    def log_message(self, *args):
        pass


def selftest():
    md = "meta\n## Page 1\ncover\n## Page 2\ntoc\n## Page 3\nch1\n## Page 4\nmore\n## Page 5\nch2\n"
    pages = split_pages(md)
    assert [no for no, _ in pages] == [1, 2, 3, 4, 5] and pages[0][1].startswith("meta"), pages
    plan = plan_parts(pages, [{"title": "하나", "start": 3}, {"title": "둘", "start": 5}])
    assert plan == [("앞부분", 1, 2), ("하나", 3, 4), ("둘", 5, 5)], plan
    assert plan_parts(pages, [{"title": "a", "start": 1}]) == [("a", 1, 5)]
    for bad in ([], [5, 3], [3, 3], [0, 3], [3, 6]):
        try:
            plan_parts(pages, [{"title": "x", "start": s} for s in bad])
        except ValueError:
            continue
        raise AssertionError(bad)
    assert file_name(1, 'a/b:c  "d".md') == "01_a_b_c_d_md.md"
    assert file_name(2, "..\\..\\제곱근과 실수") == "02_제곱근과_실수.md"
    assert head("## Page 1\n\n*[Image OCR]\n![x](y.png) 제목\n  본문") == "제목 본문"


def main():
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        print("selftest ok")
        return
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()  # one at a time: keeps API use predictable
    print(f"http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
