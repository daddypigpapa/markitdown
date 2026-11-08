"""Convert a PDF with markitdown-ocr, either as documented (plugin) or full-page.

The plugin only OCRs full pages when the PDF has no text layer at all; textbooks
whose math is drawn with symbol fonts have a broken text layer, so "full" calls
the plugin's own _ocr_full_pages() directly. The README's CLI flags
(--llm-client/--llm-model) do not exist in this CLI, hence a script.

setup:  pip install -e packages/markitdown-ocr openai
key:    VISION_OCR_GEMINI_API_KEY from the environment, else from ~/claude/key.env
usage:  python -X utf8 scripts/ocr_pdf.py {full|plugin} IN.pdf OUT.md [first-last pages, 1-based]
"""
import functools
import io
import os
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv
from openai import OpenAI
from markitdown import MarkItDown, StreamInfo
from markitdown_ocr._ocr_service import LLMVisionOCRService
from markitdown_ocr._pdf_converter_with_ocr import PdfConverterWithOCR

# thinking off for transcription: 3-page test (any2mdhub docs/04 §7.1) -- same text, 3-14x faster, and the
# default thinking rewrote a printed problem (2x²+1/x²-2x -> x²+1/x²). Chapter splitting keeps the default.
OCR_EFFORT = "none"
MODEL = "gemini-3.8-flash"  # pinned: *-latest aliases can change underneath; 0 misreads in a 3-page comparison
PROMPT = (
    "This is one page of a Korean middle-school math textbook. "
    "Transcribe everything on the page as GitHub-flavored Markdown, in reading order. "
    "Copy exactly as printed: do not solve, correct, complete or comment on anything, and add no notes or guesses. "
    "On a two-column page, read the whole left column top to bottom, then the right column. "
    "If a character cannot be read, write [?] in its place instead of guessing. "
    "Write all math in LaTeX: inline as $...$, display as $$...$$ "
    "(square roots as \\sqrt{}, powers as ^{}, fractions as \\frac{}{}). "
    "Keep Korean text exactly as printed, with normal Korean word spacing. "
    "Keep problem numbers, circled numbers and answer choices. Use Markdown tables for tables. "
    "Describe a figure or graph in one short line in square brackets. "
    "Omit the printer's slug line (file name, date, '페이지', 'MAC3') and decorative elements. "
    "Return only the Markdown, no commentary."
)


class LoggedOCR(LLMVisionOCRService):
    """The plugin turns API errors into empty text; log them and retry instead."""

    calls = 0

    def extract_text(self, image_stream, **kwargs):
        LoggedOCR.calls += 1
        for attempt in range(1, 5):
            r = super().extract_text(image_stream, **kwargs)
            if not r.error:
                print(f"  call {LoggedOCR.calls}: {len(r.text)} chars", file=sys.stderr, flush=True)
                return r
            print(f"  call {LoggedOCR.calls} error (try {attempt}): {r.error[:200]}", file=sys.stderr, flush=True)
            time.sleep(15 * attempt)
        return r


def load_pdf(path, pages):
    if not pages:
        return io.BytesIO(Path(path).read_bytes())
    first, last = map(int, pages.split("-"))
    src, dst = pdfium.PdfDocument(path), pdfium.PdfDocument.new()
    dst.import_pages(src, list(range(first - 1, last)))
    buf = io.BytesIO()
    dst.save(buf)
    return buf


def make_client(reasoning_effort=None):
    load_dotenv(Path.home() / "claude" / "key.env")  # does not override a real env var
    client = OpenAI(
        api_key=os.environ["VISION_OCR_GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    if reasoning_effort:  # the OCR plugin calls create() itself, so the setting rides on the client
        client.chat.completions.create = functools.partial(client.chat.completions.create, reasoning_effort=reasoning_effort)
    return client


def main():
    mode, src, out = sys.argv[1:4]
    pages = sys.argv[4] if len(sys.argv) > 4 else None
    client = make_client(OCR_EFFORT)
    pdf = load_pdf(src, pages)
    t0 = time.time()
    if mode == "full":
        service = LoggedOCR(client=client, model=MODEL, default_prompt=PROMPT)
        markdown = PdfConverterWithOCR()._ocr_full_pages(pdf, service)
    else:  # as documented: MarkItDown(enable_plugins=True, llm_client, llm_model)
        md = MarkItDown(enable_plugins=True, llm_client=client, llm_model=MODEL)
        markdown = md.convert_stream(pdf, stream_info=StreamInfo(extension=".pdf")).markdown
    Path(out).write_text(markdown, encoding="utf-8")
    print(f"done {mode} {Path(src).name} pages={pages or 'all'} -> {len(markdown)} chars in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
