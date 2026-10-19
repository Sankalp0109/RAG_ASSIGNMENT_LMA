"""PDF loading + article-aware chunking for the F1 Sporting Regulations corpus."""
import re
import fitz  # PyMuPDF

HEADER_FOOTER = [
    re.compile(r"^\s*2024 Formula 1 Sporting Regulations\s+\d+/\d+\s+30 April 2024\s*$"),
    re.compile(r"^\s*©2024 Fédération Internationale de l.Automobile\s+Issue 6\s*$"),
    re.compile(r"^\s*2024 Formula 1 Sporting Regulations\s*$"),
    re.compile(r"^\s*\d+/106\s*$"),
    re.compile(r"^\s*30 April 2024\s*$"),
    re.compile(r"^\s*©2024 Fédération Internationale de l.Automobile\s*$"),
    re.compile(r"^\s*Issue 6\s*$"),
]
ARTICLE_RE = re.compile(r"^(\d{1,2})\)$")          # PyMuPDF puts "36)" on its own line, title on the next
SUBART_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\s+(.*))?$")  # "36.1" alone, or "26.13 Unless authorised ..."
APPENDIX_RE = re.compile(r"^APPENDIX \d+")
MARKER_RE = re.compile(r"^(\d{1,2}\.\d{1,2}|([a-z] )?[a-z]{1,2}\)|[ivx]+\))$")


def load_pages(pdf_path):
    """Return a list of (page_number, cleaned_text) with headers/footers removed."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        lines = []
        for line in page.get_text("text").splitlines():
            if any(p.match(line) for p in HEADER_FOOTER):
                continue
            lines.append(line.rstrip())
        pages.append((i + 1, "\n".join(lines)))
    return pages


def split_long(text, max_chars, overlap):
    """Split an over-long unit on sentence/line boundaries with a small overlap.
    Returns (start, end, piece) triples so each piece can be mapped back to its pages."""
    pieces, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        pieces.append((start, end, text[start:end].strip()))
        if end >= len(text):
            break
        # start the overlap on a word boundary so the next chunk never begins mid-word
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)
        start = space + 1 if space != -1 else nxt
    return pieces


def chunk_document(pdf_path, max_chars=900, overlap=150, min_chars=200, skip_pages=1):
    """Article-aware chunking.

    1. Walk the text line by line, tracking the current top-level article ("36) REFUELLING")
       and the current sub-article number ("36.3").
    2. Each sub-article becomes one unit; tiny consecutive units of the same article are merged,
       over-long units are split with overlap.
    3. Every chunk keeps metadata: article number/title, sub-article ids and the pages it spans.
    """
    pages = load_pages(pdf_path)[skip_pages:]  # page 1 = table of contents
    lines = [(pg, l.strip()) for pg, text in pages for l in text.splitlines() if l.strip()]
    units, cur = [], None
    art_no, art_title, last_sub = None, None, 0
    i = 0
    while i < len(lines):
        page_no, s = lines[i]
        m_art, m_sub = ARTICLE_RE.match(s), SUBART_RE.match(s)
        # "54.3" only counts as a heading if it belongs to the current article and the
        # numbering moves forward (otherwise it is a cross-reference inside a sentence)
        if m_sub and (m_sub.group(1) != art_no or int(m_sub.group(2)) <= last_sub):
            m_sub = None
        if m_art and i + 1 < len(lines) and lines[i + 1][1].isupper():
            if cur:
                units.append(cur)
            art_no, art_title = m_art.group(1), lines[i + 1][1].title()
            cur, i, last_sub = None, i + 2, 0
            continue
        if APPENDIX_RE.match(s):
            if cur:
                units.append(cur)
            art_no, art_title = "App" + s.split()[1].rstrip(":"), s.title()
            cur = {"article": art_no, "title": art_title, "subs": [], "marks": [(0, page_no)], "text": s}
        elif m_sub:
            if cur:
                units.append(cur)
            sub_id = f"{m_sub.group(1)}.{m_sub.group(2)}"
            last_sub = int(m_sub.group(2))
            cur = {"article": art_no, "title": art_title, "subs": [sub_id], "marks": [(0, page_no)], "text": s}
        elif cur is not None:
            # glue list markers such as "a)" / "ii)" / "36.1" to the sentence that follows them
            last_line = cur["text"].rsplit("\n", 1)[-1]
            cur["text"] += (" " if MARKER_RE.match(last_line) else "\n")
            if page_no != cur["marks"][-1][1]:
                cur["marks"].append((len(cur["text"]), page_no))  # page break inside this unit
            cur["text"] += s
        i += 1
    if cur:
        units.append(cur)

    # merge tiny neighbours that belong to the same article
    merged = []
    for u in units:
        if merged and merged[-1]["article"] == u["article"] and \
                len(merged[-1]["text"]) < min_chars and len(merged[-1]["text"]) + len(u["text"]) <= max_chars:
            offset = len(merged[-1]["text"]) + 1
            merged[-1]["text"] += "\n" + u["text"]
            merged[-1]["subs"] += u["subs"]
            merged[-1]["marks"] += [(o + offset, p) for o, p in u["marks"]]
        else:
            merged.append(dict(u, subs=list(u["subs"]), marks=list(u["marks"])))

    chunks = []
    for u in merged:
        for start, end, part in split_long(u["text"], max_chars, overlap):
            # pages = page active at `start` plus any page that begins inside [start, end)
            pages = [p for o, p in u["marks"] if o <= start][-1:] + [p for o, p in u["marks"] if start < o < end]
            chunks.append({
                "chunk_id": len(chunks),
                "article": u["article"],
                "title": u["title"],
                "subs": u["subs"],
                "pages": sorted(set(pages)),
                "text": part,
            })
    return chunks
