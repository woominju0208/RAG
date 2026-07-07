import re
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_core.documents import Document

SKIP_TAGS = {
    "INSERTION", "COMMENT", "LIBRARYLIST", "LIBRARY", "PGBRK", "TOC",
    "APPENDIX", "IMAGE", "IMG", "IMG-CAPTION", "TITLE", "SUMMARY", "EXTRACTION",
}
SECTION_RE = re.compile(r"^SECTION-\d+$")
CELL_TAGS = ("TD", "TU", "TE", "TH")

_BARE_AMP_RE = re.compile(r"&(?!cr;|amp;|lt;|gt;|quot;|apos;|#)")
_VALID_TAG_RE = re.compile(
    r'^<(?:\?xml[^?]*\?'
    r'|/[A-Za-z][A-Za-z0-9\-]*\s*'
    r'|[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z_:][A-Za-z0-9_\-:.]*="[^"]*")*\s*/?'
    r')>$'
)


def _escape_stray_brackets(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char == "<":
            close = text.find(">", i + 1)
            next_open = text.find("<", i + 1)
            if close == -1 or (next_open != -1 and next_open < close):
                out.append("&lt;")
                i += 1
                continue
            fragment = text[i:close + 1]
            if _VALID_TAG_RE.match(fragment):
                out.append(fragment)
            else:
                out.append(fragment.replace("<", "&lt;").replace(">", "&gt;"))
            i = close + 1
        elif char == ">":
            out.append("&gt;")
            i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _render_text(elem: ET.Element) -> str:
    return _clean_text("".join(elem.itertext()))


def _render_table(table_elem: ET.Element) -> str:
    rows = []
    for tr in table_elem.iter("TR"):
        cells = [c for c in tr if c.tag in CELL_TAGS]
        cell_texts = [_render_text(c) for c in cells]
        if any(cell_texts):
            rows.append(" | ".join(cell_texts))
    return "\n".join(rows)


def _render_node(elem: ET.Element) -> str:
    if elem.tag in SKIP_TAGS:
        return ""
    if elem.tag == "P":
        return _render_text(elem)
    if elem.tag == "TABLE":
        return _render_table(elem)

    parts = [_render_node(child) for child in elem]
    combined = "\n\n".join(p for p in parts if p)
    return combined


def _walk_section(elem: ET.Element, breadcrumb: list[str], out: list[tuple[list[str], str]]) -> None:
    title_elem = elem.find("TITLE")
    title = _render_text(title_elem) if title_elem is not None else ""
    current_breadcrumb = breadcrumb + [title] if title else breadcrumb

    own_parts = []
    child_sections = []
    for child in elem:
        if child.tag == "TITLE":
            continue
        if SECTION_RE.match(child.tag):
            child_sections.append(child)
        else:
            text = _render_node(child)
            if text:
                own_parts.append(text)

    own_text = "\n\n".join(own_parts).strip()
    if own_text:
        out.append((current_breadcrumb, own_text))

    for child_section in child_sections:
        _walk_section(child_section, current_breadcrumb, out)


def parse_dart_xml(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8")
    raw = _BARE_AMP_RE.sub("&amp;", raw)
    raw = _escape_stray_brackets(raw)
    raw = raw.replace("&cr;", "\n")
    root = ET.fromstring(raw)

    # DOCUMENT-NAME/COMPANY-NAME은 파일에 따라 DOCUMENT-HEADER로 감싸져 있거나
    # DOCUMENT 바로 아래 있거나 스키마가 갈리므로 위치에 상관없이 찾는다.
    doc_name_elem = root.find(".//DOCUMENT-NAME")
    company_elem = root.find(".//COMPANY-NAME")
    doc_type = _render_text(doc_name_elem) if doc_name_elem is not None else ""
    company = _render_text(company_elem) if company_elem is not None else ""

    body = root.find("BODY")
    if body is None:
        return []

    sections: list[tuple[list[str], str]] = []
    for child in body:
        if child.tag == "COVER":
            sections.append((["표지"], _render_node(child)))
        elif SECTION_RE.match(child.tag):
            _walk_section(child, [], sections)

    documents = []
    for breadcrumb, text in sections:
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "company": company,
                    "doc_type": doc_type,
                    "section": " > ".join(breadcrumb),
                },
            )
        )
    return documents
