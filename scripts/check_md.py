"""Flag pages of an OCR'd Markdown file (ocr_pdf.py output) that need a human look.

Cheap, deterministic checks for the misreads an LLM transcriber is known to make.
They catch symptoms, not correctness: a page that passes can still be wrong.

usage:  python -X utf8 scripts/check_md.py FILE.md [...]     (no args: self-test)
exit:   1 if any page was flagged
"""
import re
import sys
from pathlib import Path

PAGE = re.compile(r"^## Page (\d+)\n(.*?)(?=^## Page |\Z)", re.S | re.M)
COMMENTARY = re.compile(r"\[참고|일 수 있음|\(추정\)|\[Note")  # notes the source never has
UNREADABLE = "[?]"
# ponytail: a lone $\dots$ right after a problem number is filler; \cdots inside formulas is legit
FILLER = re.compile(r"^\**\d+(?:-\d+)?\**\s*\$\\c?dots\$", re.M)
# ponytail: only "N-M" numbering (1-1, 1-2, 2-1 ...); bold "**5**" style numbers are not tracked
LABEL = re.compile(r"^(?:#+\s*)?\**(\d+)-(\d+)\**(?=[\s○●]|$)", re.M)


def pages(text):
    for m in PAGE.finditer(text):
        yield int(m.group(1)), m.group(2).replace("*[Image OCR]", "").replace("[End OCR]*", "")


def check(text):
    """Return [(page, reason), ...] for one file."""
    out, prev = [], None
    for no, body in pages(text):
        if len(body.strip()) < 20:
            out.append((no, "empty page"))
        if m := COMMENTARY.search(body):
            out.append((no, f"commentary {m.group()!r}"))
        if UNREADABLE in body:
            out.append((no, f"unreadable [?] x{body.count(UNREADABLE)}"))
        if FILLER.search(body):
            out.append((no, "filler $\\dots$ after a problem number"))
        if body.replace("\\$", "").count("$") % 2:
            out.append((no, "odd number of $"))
        if (o := len(re.findall(r"(?<!\\)\{", body))) != (c := len(re.findall(r"(?<!\\)\}", body))):
            out.append((no, f"braces {{ {o} vs }} {c}"))
        for n, k in ((int(a), int(b)) for a, b in LABEL.findall(body)):
            # next in section, next section, or a new chapter starting over at 1-1
            if prev and not ((n == prev[0] and k == prev[1] + 1) or (n in (prev[0] + 1, 1) and k == 1)):
                out.append((no, f"numbering {prev[0]}-{prev[1]} -> {n}-{k}"))
            prev = (n, k)
    return out


def selftest():
    good = "## Page 1\n*[Image OCR]\n1-1 다음 식을 인수분해하여라. $x^{2}+\\cdots+1$\n1-2 ...\n## Page 2\n2-1 $a$ 와 $b$ 의 값을 구하여라.\n[End OCR]*\n"
    assert check(good) == [], check(good)
    bad = ("## Page 1\n1-1 $x^{2$ 의 값\n[참고: 5(x+y)일 수 있음]\n1-3 $\\dots$\n"
           "## Page 2\n**2-1** $\\dots$ 원문 [?] 글자\n## Page 3\n\n")
    reasons = [r.split()[0] for _, r in check(bad)]
    assert reasons == ["commentary", "filler", "braces", "numbering", "unreadable", "filler", "empty"], reasons


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
        print("selftest ok")
        sys.exit(0)
    flagged = 0
    for path in sys.argv[1:]:
        for no, reason in check(Path(path).read_text(encoding="utf-8")):
            print(f"{Path(path).name}  Page {no}: {reason}")
            flagged += 1
    print(f"{flagged} flag(s)", file=sys.stderr)
    sys.exit(1 if flagged else 0)
