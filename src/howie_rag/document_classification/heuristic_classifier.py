import re
from typing import Dict, List


TABLE_KEYWORDS = {
    "table",
    "tabelle",
    "variable",
    "codebook",
    "indikator",
    "indicator",
    "dataset",
    "year",
    "unit",
    "quelle",
    "source",
    "csv",
}

FIGURE_KEYWORDS = {
    "figure",
    "fig.",
    "grafik",
    "abbildung",
    "chart",
}

NARRATIVE_KEYWORDS = {
    "report",
    "overview",
    "introduction",
    "method",
    "methodology",
    "results",
    "discussion",
    "survey",
    "study",
    "analysis",
}


def _non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _digit_ratio(text: str) -> float:
    non_space_characters = [character for character in text if not character.isspace()]
    if not non_space_characters:
        return 0.0
    digit_count = sum(1 for character in non_space_characters if character.isdigit())
    return digit_count / len(non_space_characters)


def _table_like_line(line: str) -> bool:
    if line.count(";") >= 3 or line.count("\t") >= 3:
        return True
    numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", line)
    return len(numeric_tokens) >= 4


def _starts_with_numbered_heading(line: str) -> bool:
    index = 0
    length = len(line)
    saw_digit_group = False

    while index < length:
        digit_start = index
        while index < length and line[index].isdigit():
            index += 1

        if index == digit_start:
            return False

        saw_digit_group = True
        if index >= length:
            return False

        if line[index].isspace():
            return saw_digit_group

        if line[index] != ".":
            return False

        index += 1
        if index >= length:
            return False

        if line[index].isspace():
            return saw_digit_group


def _heading_like_line(line: str) -> bool:
    if len(line) > 100:
        return False
    if line.endswith(":"):
        return True
    if _starts_with_numbered_heading(line):
        return True
    if line.startswith("#"):
        return True
    words = line.split()
    if 1 <= len(words) <= 10 and line == line.title():
        return True
    return False


def _paragraph_like_line(line: str) -> bool:
    return len(line) >= 80 and any(punctuation in line for punctuation in ".!?")


def _keyword_hits(text: str, keywords: set) -> int:
    lowered_text = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered_text)


def classify_document_content(title: str, text: str, file_type: str) -> Dict[str, object]:
    lines = _non_empty_lines(text)
    line_count = len(lines)
    average_line_length = sum(len(line) for line in lines) / line_count if line_count else 0.0

    table_like_line_count = sum(1 for line in lines if _table_like_line(line))
    heading_like_line_count = sum(1 for line in lines if _heading_like_line(line))
    paragraph_like_line_count = sum(1 for line in lines if _paragraph_like_line(line))

    table_like_line_ratio = table_like_line_count / line_count if line_count else 0.0
    heading_like_line_ratio = heading_like_line_count / line_count if line_count else 0.0
    paragraph_like_line_ratio = paragraph_like_line_count / line_count if line_count else 0.0
    digit_ratio = _digit_ratio(text)

    title_and_text = f"{title}\n{text}"
    table_keyword_hits = _keyword_hits(title_and_text, TABLE_KEYWORDS)
    figure_keyword_hits = _keyword_hits(title_and_text, FIGURE_KEYWORDS)
    narrative_keyword_hits = _keyword_hits(title_and_text, NARRATIVE_KEYWORDS)

    has_tables = table_like_line_ratio >= 0.2 or table_keyword_hits >= 2 or file_type in {"csv", "xls"}
    has_figures = figure_keyword_hits > 0

    statistical_score = 0
    narrative_score = 0

    if digit_ratio >= 0.15:
        statistical_score += 2
    elif digit_ratio >= 0.08:
        statistical_score += 1

    if table_like_line_ratio >= 0.3:
        statistical_score += 3
    elif table_like_line_ratio >= 0.15:
        statistical_score += 2
    elif table_like_line_ratio >= 0.05:
        statistical_score += 1

    if table_keyword_hits >= 3:
        statistical_score += 2
    elif table_keyword_hits >= 1:
        statistical_score += 1

    if file_type in {"csv", "xls"}:
        statistical_score += 2

    if paragraph_like_line_ratio >= 0.2:
        narrative_score += 2
    elif paragraph_like_line_ratio >= 0.08:
        narrative_score += 1

    if heading_like_line_ratio >= 0.08:
        narrative_score += 1

    if average_line_length >= 70:
        narrative_score += 1

    if narrative_keyword_hits >= 3:
        narrative_score += 2
    elif narrative_keyword_hits >= 1:
        narrative_score += 1

    if statistical_score >= 3 and narrative_score >= 3:
        document_type = "mixed"
    elif statistical_score > narrative_score:
        document_type = "statistical"
    else:
        document_type = "narrative"

    return {
        "document_type": document_type,
        "has_tables": has_tables,
        "has_figures": has_figures,
        "classification_stats": {
            "digit_ratio": round(digit_ratio, 4),
            "line_count": line_count,
            "avg_line_length": round(average_line_length, 2),
            "table_like_line_ratio": round(table_like_line_ratio, 4),
            "heading_like_line_ratio": round(heading_like_line_ratio, 4),
            "paragraph_like_line_ratio": round(paragraph_like_line_ratio, 4),
            "table_keyword_hits": table_keyword_hits,
            "figure_keyword_hits": figure_keyword_hits,
            "narrative_keyword_hits": narrative_keyword_hits,
            "statistical_score": statistical_score,
            "narrative_score": narrative_score,
        },
    }
