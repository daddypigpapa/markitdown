"""Convert a PDF with markitdown-ocr, either as documented (plugin) or full-page.

The plugin only OCRs full pages when the PDF has no text layer at all; textbooks
whose math is drawn with symbol fonts have a broken text layer, so "full" calls
the plugin's own _ocr_full_pages() directly. The README's CLI flags
(--llm-client/--llm-model) do not exist in this CLI, hence a script.

setup:  pip install -e packages/markitdown-ocr openai
key:    VISION_OCR_GEMINI_API_KEY from the environment, else from ~/claude/key.env
usage:  python -X utf8 scripts/ocr_pdf.py {full|plugin} IN.pdf OUT.md [first-last pages, 1-based] [--kinds=figure,math,text]
        (--kinds picks the full-mode prompt by document type; see KINDS)
"""
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
# --kinds: what the uploader said the document is (any2mdhub's upload checkboxes). Without it, PROMPT above.
CORE = (
    "This is one page of a document. "
    "Transcribe everything on the page as GitHub-flavored Markdown, in reading order. "
    "Copy exactly as printed: do not solve, correct, complete or comment on anything, and add no notes or guesses. "
    "On a multi-column page, read the whole left column top to bottom, then the next column. "
    "If a character cannot be read, write [?] in its place instead of guessing. "
    "Keep Korean text exactly as printed, with normal Korean word spacing. Use Markdown tables for tables. "
    "Omit running headers and footers, page numbers, the printer's slug line and decorative elements. "
    "Return only the Markdown, no commentary."
)
KINDS = {
    "figure": "The page mixes pictures and text. For each photo, illustration or diagram, write one line in square "
              "brackets saying what it shows, at the place it appears, and keep its caption.",
    "math": "The page is mostly math and graphs. Write all math in LaTeX: inline as $...$, display as $$...$$ "
            "(square roots as \\sqrt{}, powers as ^{}, fractions as \\frac{}{}). Keep problem numbers, circled numbers "
            "and answer choices. For a graph, write one line in square brackets with its axes, labels and marked points.",
    "text": "The page is mostly text. Keep headings as Markdown headings, and keep paragraphs and lists. "
            "Use LaTeX only for formulas that are actually printed.",
}


def make_prompt(kinds):
    """CORE plus one sentence group per checked kind, in a fixed order; no kinds -> PROMPT."""
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}; use {list(KINDS)}")
    return " ".join([CORE] + [KINDS[k] for k in KINDS if k in kinds]) if kinds else PROMPT


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
    real = client.chat.completions.create

    def create(*args, **kwargs):  # the OCR plugin calls create() itself, so these ride on the client
        if reasoning_effort:
            kwargs.setdefault("reasoning_effort", reasoning_effort)
        r = real(*args, **kwargs)
        if r.usage:  # Gemini's OpenAI layer leaves thinking out of completion_tokens; total - prompt is what is billed as output
            print(f"  usage model={kwargs.get('model')} in={r.usage.prompt_tokens} out={r.usage.total_tokens - r.usage.prompt_tokens}",
                  file=sys.stderr, flush=True)
        return r

    client.chat.completions.create = create
    return client


def main():
    kinds = [k for a in sys.argv if a.startswith("--kinds=") for k in a[8:].split(",") if k]
    args = [a for a in sys.argv[1:] if not a.startswith("--kinds=")]
    mode, src, out = args[:3]
    pages = args[3] if len(args) > 3 else None
    prompt = make_prompt(kinds)  # before any API call: a typo in --kinds fails here, for free
    client = make_client(OCR_EFFORT)
    pdf = load_pdf(src, pages)
    t0 = time.time()
    if mode == "full":
        service = LoggedOCR(client=client, model=MODEL, default_prompt=prompt)
        markdown = PdfConverterWithOCR()._ocr_full_pages(pdf, service)
    else:  # as documented: MarkItDown(enable_plugins=True, llm_client, llm_model)
        md = MarkItDown(enable_plugins=True, llm_client=client, llm_model=MODEL)
        markdown = md.convert_stream(pdf, stream_info=StreamInfo(extension=".pdf")).markdown
    Path(out).write_text(markdown, encoding="utf-8")
    print(f"done {mode} {Path(src).name} pages={pages or 'all'} -> {len(markdown)} chars in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
