from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.document_classification import classify_document_content


def test_classify_document_content_detects_narrative_text() -> None:
    text = (
        "Introduction\n"
        "This report explains the methodology of the student survey and discusses the results in detail. "
        "The study analysis describes how the survey was conducted and how the findings should be interpreted."
    )

    result = classify_document_content("Methods Report", text, "pdf")

    assert result["document_type"] == "narrative"
    assert result["has_tables"] is False


def test_classify_document_content_detects_statistical_text() -> None:
    text = (
        "Tab.0.1;;;;;;;;\n"
        "2019;2020;2021;2022;2023;2024\n"
        "Studierende;2891;2944;2946;2920;2868;2864\n"
        "Männlich;1465;1476;1468;1444;1408;1399\n"
        "Weiblich;1426;1468;1478;1476;1460;1465\n"
    )

    result = classify_document_content("Strukturdaten", text, "csv")

    assert result["document_type"] == "statistical"
    assert result["has_tables"] is True


def test_classify_document_content_detects_mixed_text_and_figures() -> None:
    text = (
        "Figure 1: Student mobility trend\n"
        "This report discusses the student survey and explains the changes in mobility over time.\n"
        "2019;2020;2021;2022\n"
        "Mobility;12;13;14;15\n"
    )

    result = classify_document_content("Mobility Report", text, "pdf")

    assert result["document_type"] == "mixed"
    assert result["has_tables"] is True
    assert result["has_figures"] is True
