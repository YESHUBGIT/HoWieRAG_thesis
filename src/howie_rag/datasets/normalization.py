from typing import Any, Dict, Optional


def _normalized_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def normalize_source_metadata(
    *,
    dataset_name: str,
    domain: str,
    context_id: str,
    title: str = "",
    source_file: str = "",
    subset: str = "",
    split: str = "",
    source_year: str = "",
    source_entity: str = "",
    source_page_number: str = "",
    source_section: str = "",
) -> Dict[str, object]:
    normalized: Dict[str, object] = {
        "dataset_name": dataset_name,
        "domain": domain,
        "context_id": context_id,
        "source_title": _normalized_string(title) or _normalized_string(context_id),
        "source_file": _normalized_string(source_file),
        "source_domain": _normalized_string(domain),
    }

    if subset:
        normalized["source_subset"] = _normalized_string(subset)
    if split:
        normalized["source_split"] = _normalized_string(split)
    if source_year:
        normalized["source_year"] = _normalized_string(source_year)
    if source_entity:
        normalized["source_entity"] = _normalized_string(source_entity)
    if source_page_number:
        normalized["source_page_number"] = _normalized_string(source_page_number)
    if source_section:
        normalized["source_section"] = _normalized_string(source_section)

    return normalized
