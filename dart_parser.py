import re
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_core.documents import Document

SKIP_TAGS = {
    "INSERTION", "COMMENT", "LIBRARYLIST", "PGBRK", "TOC",
    "APPENDIX", "IMAGE", "IMG", "IMG-CAPTION", "TITLE", "SUMMARY", "EXTRACTION",
}
# LIBRARY는 예전엔 SKIP_TAGS에 있었는데, 재무상태표/손익계산서 같은 실제 재무제표
# TABLE-GROUP이 LIBRARY 태그 안에 들어있는 경우가 82개 파일 전부에서 발견됐다(파일당
# 최대 40개). LIBRARY를 건너뛰면 이 표들이 통째로 사라져서 "유동자산 얼마야?" 같은
# 질문에 답을 못 찾았다. LIBRARY 자체에는 렌더링되는 내용이 없는 FILENAME/INSERTION만
# 있는 경우도 있지만 그건 그대로 빈 텍스트로 처리되니 안전하다.
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


# 행 사이의 경계를 일반 "\n"으로 표시하면, 셀 안에 원래 있던 줄바꿈(긴 법률 조항 등 여러 줄
# 문단이 든 셀)과 구분이 안 돼서 진짜 행 경계가 아닌 곳에서 표가 잘리는 문제가 있다. 그래서
# 행 경계만 이 문자로 표시해두고, ingest.py가 표를 자를지 정할 때만 이 경계로 나누고
# 최종 저장 시에는 다시 "\n"으로 되돌린다.
TABLE_ROW_SEP = "\x1f"


def _render_table(table_elem: ET.Element) -> str:
    rows = []
    for tr in table_elem.iter("TR"):
        cells = [c for c in tr if c.tag in CELL_TAGS]
        cell_texts = [_render_text(c) for c in cells]
        if any(cell_texts):
            rows.append(" | ".join(cell_texts))
    return TABLE_ROW_SEP.join(rows)


def _render_node(elem: ET.Element) -> str:
    """표를 문단과 합쳐 하나의 평범한 문자열로 만든다 (COVER 등에서 사용). _render_blocks와
    달리 표를 따로 떼지 않으므로, TABLE_ROW_SEP을 바로 일반 줄바꿈으로 되돌려서 특수문자가
    최종 텍스트에 남지 않게 한다."""
    if elem.tag in SKIP_TAGS:
        return ""
    if elem.tag == "P":
        return _render_text(elem)
    if elem.tag == "TABLE":
        return _render_table(elem).replace(TABLE_ROW_SEP, "\n")

    parts = [_render_node(child) for child in elem]
    combined = "\n\n".join(p for p in parts if p)
    return combined


def _render_blocks(elem: ET.Element) -> list[tuple[bool, str]]:
    """_render_node와 달리 표를 문단과 합치지 않고 (is_table, text) 블록으로 따로 뗀다.
    표가 TABLE-GROUP 등 다른 태그 안에 중첩돼 있어도 재귀적으로 찾아서 뗀다. 이러면
    ingest.py에서 표는 셀 중간을 자르지 않고 행 단위로, 문단은 지금처럼 글자 수로
    나눌 수 있다."""
    if elem.tag in SKIP_TAGS:
        return []
    if elem.tag == "P":
        text = _render_text(elem)
        return [(False, text)] if text else []
    if elem.tag == "TABLE":
        text = _render_table(elem)
        return [(True, text)] if text else []

    blocks = []
    for child in elem:
        blocks.extend(_render_blocks(child))
    return blocks


SMALL_TABLE_MERGE_LIMIT = 200  # 이 크기 이하인 표는 따로 안 떼고 주변 문단과 합친다
CAPTION_MERGE_LIMIT = 100  # 표 바로 앞의 이 크기 이하 조각(단위 표시 등)은 표에 붙인다
CAPTION_MERGE_TOTAL_LIMIT = 150  # 캡션 여러 개를 합쳐 붙일 때의 총량 상한


def _merge_text_blocks(blocks: list[tuple[bool, str]]) -> list[tuple[bool, str]]:
    """표가 아닌 블록끼리는 다시 하나로 합쳐서(지금까지처럼) text_splitter가 문단
    경계를 존중하며 글자 수 기준으로 나눌 수 있게 하고, 표 블록은 독립적으로 남긴다.

    다만 아주 작은 표(SMALL_TABLE_MERGE_LIMIT 이하, 예: "일자 | 주소 | 비고" 몇 글자짜리)까지
    전부 떼어내면 청크 수가 크게 늘고, 그 표만 뚝 떨어져서 앞뒤 맥락 없이 검색되는 문제가
    있었다. 그래서 작은 표는 표 취급을 하지 않고 문단처럼 합친다 — 이때 표 안에서만 쓰던
    행 경계 표시(TABLE_ROW_SEP)는 더 이상 필요 없으니 일반 줄바꿈으로 되돌린다.

    표 바로 앞의 짧은 문단(예: "(단위 : 명, 시간)")은 표 자체를 소개하는 캡션인 경우가
    대부분이라, 독립된 청크로 떼어놓으면 실제 데이터 없이 헤더/캡션만 있는 청크가 생겨서
    검색에서 오히려 진짜 데이터가 있는 표 청크보다 이겨버리는 문제가 있었다. 그래서 표
    바로 앞의 짧은 버퍼는 별도 청크로 내보내지 않고 표 앞에 붙인다.

    이때 "버퍼 전체 길이"가 아니라 버퍼 끝에서부터 짧은 조각을 계속 모은 것만 본다.
    "단위 :백만원" 같은 캡션은 P가 아니라 테두리 없는 1칸짜리 TABLE로 렌더링된 경우도
    있어서(그래서 is_table=True인데도 아주 짧음), "5. 매출에 관한 사항" / "가. 매출실적" /
    "(단위 :백만원)"처럼 짧은 조각이 표 바로 앞에 **여러 개 연달아** 쌓이기도 한다. 마지막
    한 조각만 보면 그중 표에 진짜 붙어야 할 라벨("가. 매출실적")을 놓치고 캡션만 붙이게
    되는 문제가 있었다. 그래서 끝에서부터 짧은 조각을 계속 꺼내며 모으다가, 앞의 조각이
    무관한 긴 문단(설비투자현황 등)이면 거기서 멈춘다."""
    merged = []
    buffer = []
    for is_table, text in blocks:
        if is_table and len(text) > SMALL_TABLE_MERGE_LIMIT:
            caption_parts: list[str] = []
            caption_len = 0
            while buffer and len(buffer[-1]) <= CAPTION_MERGE_LIMIT and caption_len + len(buffer[-1]) <= CAPTION_MERGE_TOTAL_LIMIT:
                piece = buffer.pop()
                caption_parts.insert(0, piece)
                caption_len += len(piece)
            if caption_parts:
                text = "\n\n".join(caption_parts) + "\n\n" + text
            if buffer:
                merged.append((False, "\n\n".join(buffer)))
                buffer = []
            merged.append((True, text))
        elif is_table:
            buffer.append(text.replace(TABLE_ROW_SEP, "\n"))
        else:
            buffer.append(text)
    if buffer:
        merged.append((False, "\n\n".join(buffer)))
    return merged


_CEO_LABEL_RE = re.compile(r"대\s*표\s*(?:이\s*사|자\s*명|자)\s*(?:명)?\s*:\s*\|?\s*([^\n|]+)")
_CEO_SIGNATURE_RE = re.compile(r"대표이사\s+([가-힣]{2,4})(?:\s|$)")


def _extract_ceo_name(sections: list[tuple[list[str], bool, str]]) -> str:
    """대표이사 이름은 문서 형식에 따라 위치가 다르다.
    - 사업보고서/반기보고서류: 표지 표에 '대표이사 : 이름' 형태로 있다. 공동대표인 경우
      '최낙현, 강호성'처럼 콤마로 여러 명이 나오기도 해서 그 줄 전체를 잡는다.
    - 감사보고서(재무제표 첨부)류: 표지엔 없고, 본문의 재무제표 책임 확인 문구에
      '회사명 대표이사 이름' 서명 형태로만 나온다.
    두 형태를 문서 전체에서 순서대로 찾아본다. (표 블록은 행 경계가 TABLE_ROW_SEP으로
    표시돼 있어서, 줄 단위 정규식이 통하도록 먼저 일반 줄바꿈으로 되돌린다.)"""
    for _, _, text in sections:
        match = _CEO_LABEL_RE.search(text.replace(TABLE_ROW_SEP, "\n"))
        if match:
            return match.group(1).strip()
    for _, _, text in sections:
        match = _CEO_SIGNATURE_RE.search(text.replace(TABLE_ROW_SEP, "\n"))
        if match:
            return match.group(1)
    return ""


def _walk_section(elem: ET.Element, breadcrumb: list[str], out: list[tuple[list[str], bool, str]]) -> None:
    title_elem = elem.find("TITLE")
    title = _render_text(title_elem) if title_elem is not None else ""
    current_breadcrumb = breadcrumb + [title] if title else breadcrumb

    own_blocks: list[tuple[bool, str]] = []
    child_sections = []
    for child in elem:
        if child.tag == "TITLE":
            continue
        if SECTION_RE.match(child.tag):
            child_sections.append(child)
        else:
            own_blocks.extend(_render_blocks(child))

    for is_table, text in _merge_text_blocks(own_blocks):
        if text.strip():
            out.append((current_breadcrumb, is_table, text))

    for child_section in child_sections:
        _walk_section(child_section, current_breadcrumb, out)


_REGNO_RE = re.compile(r"^\d{6}-\d{7}$")
_AFFILIATE_NAME_COLUMNS = {"계열회사", "기업명"}


def _extract_affiliates(sections: list[tuple[list[str], bool, str]]) -> str:
    """"계열회사 | 법인등록번호" 또는 "상장여부 | 회사수 | 기업명 | 법인등록번호" 형태의 표에서
    회사명만 뽑는다. 원본 XML이 ROWSPAN으로 상장여부/회사수 셀을 묶어놔서, 같은 표 안에서도
    첫 행은 4컬럼, 이어지는 행은 2컬럼(회사명 | 법인등록번호)으로 렌더링된다. 그래서 헤더로
    컬럼 위치를 고정하지 않고, 매 행에서 "마지막 두 컬럼 = (회사명, 법인등록번호)"로 찾는다.
    (표 블록은 행 경계가 TABLE_ROW_SEP이라 먼저 일반 줄바꿈으로 되돌린다.)"""
    for _, _, text in sections:
        lines = text.replace(TABLE_ROW_SEP, "\n").split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 2 and cols[-2] in _AFFILIATE_NAME_COLUMNS and cols[-1] == "법인등록번호":
                header_idx = i
                break
        if header_idx is None:
            continue

        names = []
        for row in lines[header_idx + 1:]:
            if "|" not in row or not row.strip():
                break
            cols = [c.strip() for c in row.split("|")]
            if len(cols) < 2:
                break
            name, regno = cols[-2], cols[-1]
            if not (regno == "-" or _REGNO_RE.match(regno)):
                break
            if name:
                names.append(name)
        if names:
            return ", ".join(dict.fromkeys(names))
    return ""


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

    # 표지의 PERIODFROM/PERIODTO(AUNITVALUE=YYYYMMDD)가 그 파일이 다루는 회계기간이다.
    # 청크마다 어떤 사업연도 자료인지 표시가 없으면, 검색된 청크만 보고는 몇 년도 것인지
    # 알 수 없어 LLM이 연도를 추측하다 틀리는 문제가 있었다.
    period_from_elem = root.find(".//TU[@AUNIT='PERIODFROM']")
    period_to_elem = root.find(".//TU[@AUNIT='PERIODTO']")
    fiscal_period = ""
    if period_from_elem is not None and period_to_elem is not None:
        from_value = period_from_elem.get("AUNITVALUE", "")
        to_value = period_to_elem.get("AUNITVALUE", "")
        if len(from_value) == 8 and len(to_value) == 8:
            fiscal_period = (
                f"{from_value[:4]}.{from_value[4:6]}.{from_value[6:]}"
                f"~{to_value[:4]}.{to_value[4:6]}.{to_value[6:]}"
            )

    body = root.find("BODY")
    if body is None:
        return []

    sections: list[tuple[list[str], bool, str]] = []
    for child in body:
        if child.tag == "COVER":
            sections.append((["표지"], False, _render_node(child)))
        elif SECTION_RE.match(child.tag):
            _walk_section(child, [], sections)

    ceo_name = _extract_ceo_name(sections)
    affiliates = _extract_affiliates(sections)

    documents = []
    for breadcrumb, is_table, text in sections:
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "company": company,
                    "doc_type": doc_type,
                    "fiscal_period": fiscal_period,
                    "ceo_name": ceo_name,
                    "is_table": is_table,
                    "affiliates": affiliates,
                    "section": " > ".join(breadcrumb),
                },
            )
        )
    return documents
