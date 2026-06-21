import argparse
import json
import re
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse


CATALOG_URL = "https://www.datenportal.bmftr.bund.de/portal/dump/govdata/metadatenkatalog.json?standard=dcatap"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
SOURCE_ROOT = REPO_ROOT / "source_data" / "govdata"

OUTPUT_DIR.mkdir(exist_ok=True)
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

RAW_PATH = OUTPUT_DIR / "metadatenkatalog.json"
LINKS_PATH = OUTPUT_DIR / "extracted_links.json"
DATASETS_PATH = OUTPUT_DIR / "datasets_enriched.json"
MANIFEST_PATH = OUTPUT_DIR / "download_manifest.json"

USER_AGENT = "HoWieRAGGovDataIngestion/1.0"

HIGH_RELEVANCE_TERMS = {
    "hochschule",
    "hochschulen",
    "hochschul",
    "studierende",
    "student",
    "students",
    "studien",
    "promotion",
    "promotionen",
    "tertiär",
    "tertiary",
    "doktoranden",
    "doktorandinnen",
}

MEDIUM_RELEVANCE_TERMS = {
    "bildung",
    "bildungs",
    "schule",
    "schulen",
    "ausbildung",
    "erwerbslos",
    "arbeitsmarkt",
    "demographie",
    "bevölkerung",
    "abschluss",
    "forschung",
    "wissenschaft",
    "isced",
}

CONTENT_SCOPE_RULES = [
    (
        "higher_education_students",
        {
            "studierende",
            "student",
            "students",
            "studienanfänger",
            "studienanfaenger",
            "hochschule",
            "hochschulen",
        },
    ),
    (
        "education_attainment",
        {
            "bildungsabschluss",
            "abschluss",
            "hochschulabschluss",
            "promotion",
            "promotionen",
            "isced",
        },
    ),
    (
        "research_and_development",
        {
            "forschung",
            "wissenschaft",
            "entwicklung",
            "hochschulpersonal",
        },
    ),
    (
        "labour_market_outcomes",
        {
            "erwerbstätige",
            "erwerbslos",
            "erwerbslosenquote",
            "arbeitsmarkt",
            "beschäftigte",
        },
    ),
    (
        "demography",
        {
            "bevölkerung",
            "alter",
            "geschlecht",
            "staatsangehörigkeit",
        },
    ),
    (
        "finance_expenditure",
        {
            "ausgaben",
            "bildungsausgaben",
            "personalausgaben",
            "bruttoinlandsprodukt",
            "steuereinnahmen",
        },
    ),
]

BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
}


def http_get_bytes(url: str) -> bytes:
    http_request = request.Request(url, headers={"User-Agent": USER_AGENT})
    with request.urlopen(http_request, timeout=120) as response:
        return response.read()


def http_get_json(url: str) -> dict:
    return json.loads(http_get_bytes(url).decode("utf-8"))


def get_id(value):
    if isinstance(value, dict):
        return value.get("@id")
    if isinstance(value, str):
        return value
    return None


def get_literal(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("@value") or value.get("@id")
    if isinstance(value, list):
        return [get_literal(v) for v in value]
    return value


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def flatten_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(flatten_to_text(item) for item in value)
    return str(value)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "resource"


def normalize_format(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.rsplit("/", 1)[-1].upper()


def build_resource_type(url: str, file_format: Optional[str]) -> str:
    normalized_format = normalize_format(file_format)
    lower_url = url.lower()

    if normalized_format == "PDF":
        return "pdf"
    if normalized_format == "CSV":
        return "csv"
    if normalized_format == "XLS":
        return "xls"
    if normalized_format == "HTML":
        if "grafik" in lower_url:
            return "graphic_html_markdown"
        return "table_html_markdown"
    return "other"


def extract_dataset_code(dataset_id: str) -> str:
    return dataset_id.rstrip("/").rsplit("/", 1)[-1]


def classify_domain_relevance(title: str, description: str, keywords: List[str], themes: List[str]) -> str:
    haystack = " ".join([title, description] + keywords + themes).lower()

    if any(term in haystack for term in HIGH_RELEVANCE_TERMS):
        return "high"
    if any(term in haystack for term in MEDIUM_RELEVANCE_TERMS) or any(theme.endswith(("EDUC", "TECH", "SOCI")) for theme in themes):
        return "medium"
    return "low"


def classify_content_scope(title: str, description: str, keywords: List[str]) -> str:
    haystack = " ".join([title, description] + keywords).lower()
    scores = []
    for scope, terms in CONTENT_SCOPE_RULES:
        score = sum(1 for term in terms if term in haystack)
        if score > 0:
            scores.append((scope, score))

    if not scores:
        return "other"

    scores.sort(key=lambda item: item[1], reverse=True)
    if len(scores) > 1 and scores[1][1] > 0:
        if scores[0][1] - scores[1][1] <= 1:
            return "mixed_education_statistics"

    return scores[0][0]


def infer_supported_intents(content_scope: str, resource_type: str) -> List[str]:
    if resource_type in {"csv", "xls", "pdf", "table_html_markdown"}:
        intents = ["FACT", "TREND_PATTERN", "COMPARISON", "SOURCE_SEEKING", "NAVIGATION"]
    else:
        intents = ["SOURCE_SEEKING", "NAVIGATION"]

    if content_scope in {"research_and_development", "education_attainment", "higher_education_students"}:
        intents.append("SUMMARY")

    seen = []
    for intent in intents:
        if intent not in seen:
            seen.append(intent)
    return seen


def classify_download_priority(domain_relevance: str, resource_type: str) -> str:
    if domain_relevance == "high" and resource_type in {"pdf", "table_html_markdown", "csv"}:
        return "high"
    if domain_relevance in {"high", "medium"} and resource_type in {"pdf", "table_html_markdown", "csv", "graphic_html_markdown"}:
        return "medium"
    if resource_type == "xls":
        return "low"
    return "skip"


def should_download(priority: str, min_priority: str) -> bool:
    order = {"skip": 0, "low": 1, "medium": 2, "high": 3}
    return order.get(priority, 0) >= order.get(min_priority, 2)


def html_to_text(html: str) -> str:
    html = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style.*?>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)

    def replace_tag(match: re.Match) -> str:
        tag = match.group(1).lower().split()[0].strip("/")
        if tag in BLOCK_TAGS:
            return "\n"
        return " "

    html = re.sub(r"<\s*/?\s*([a-zA-Z0-9]+)[^>]*>", replace_tag, html)
    text = unescape(html)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def save_html_as_markdown(destination: Path, metadata: dict, html_bytes: bytes) -> None:
    extracted_text = html_to_text(html_bytes.decode("utf-8", errors="ignore"))
    content = (
        f"---\n"
        f"dataset_id: {metadata['dataset_id']}\n"
        f"dataset_code: {metadata['dataset_code']}\n"
        f"title: {metadata['title']}\n"
        f"resource_type: {metadata['resource_type']}\n"
        f"format: {metadata['format']}\n"
        f"source_url: {metadata['url']}\n"
        f"content_scope: {metadata['content_scope']}\n"
        f"domain_relevance: {metadata['domain_relevance']}\n"
        f"---\n\n"
        f"# {metadata['title']}\n\n"
        f"{extracted_text}\n"
    )
    destination.write_text(content, encoding="utf-8")


def download_resource(record: dict, destination: Path) -> Tuple[bool, Optional[str]]:
    try:
        payload = http_get_bytes(record["url"])
        if record["resource_type"] in {"table_html_markdown", "graphic_html_markdown"}:
            save_html_as_markdown(destination, record, payload)
        else:
            destination.write_bytes(payload)
        return True, None
    except (HTTPError, URLError, TimeoutError) as error:
        return False, str(error)


def load_catalog(use_local: bool) -> dict:
    if use_local and RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text(encoding="utf-8"))

    print("Downloading metadata catalog...")
    catalog = http_get_json(CATALOG_URL)
    RAW_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def normalize_catalog(catalog: dict) -> Tuple[List[dict], List[dict]]:
    graph = catalog.get("@graph", [])
    by_id = {
        item.get("@id"): item
        for item in graph
        if isinstance(item, dict) and item.get("@id")
    }

    datasets = [
        item for item in graph
        if isinstance(item, dict) and item.get("@type") == "dcat:Dataset"
    ]

    publisher_name = get_literal(by_id.get("https://www.datenportal.bmftr.bund.de/portal/de/imprint.html#publisher", {}).get("foaf:name"))
    maintainer_name = get_literal(by_id.get("https://www.datenportal.bmftr.bund.de/portal/de/imprint.html#maintainer", {}).get("foaf:name"))

    normalized = []
    manifest = []
    seen_urls = set()

    for ds in datasets:
        dataset_id = ds.get("@id")
        dataset_code = extract_dataset_code(dataset_id)
        title = flatten_to_text(get_literal(ds.get("dcterms:title")))
        description = flatten_to_text(get_literal(ds.get("dcterms:description")))
        keywords = [item for item in as_list(get_literal(ds.get("dcat:keyword"))) if item]
        themes = [item for item in as_list(get_literal(ds.get("dcat:theme"))) if item]
        landing_page = get_id(ds.get("dcat:landingPage")) or get_id(ds.get("foaf:page"))

        domain_relevance = classify_domain_relevance(title, description, keywords, themes)
        content_scope = classify_content_scope(title, description, keywords)

        distribution_refs = as_list(ds.get("dcat:distribution"))
        resources = []

        for ref in distribution_refs:
            dist_id = get_id(ref)
            dist = by_id.get(dist_id, {}) if dist_id else {}

            access_url = get_id(dist.get("dcat:accessURL"))
            download_url = get_id(dist.get("dcat:downloadURL"))
            media_type = get_literal(dist.get("dcat:mediaType"))
            file_format = normalize_format(get_literal(dist.get("dcterms:format")))
            license_url = get_id(dist.get("dcterms:license"))

            resource_urls = []
            for url_type, url in (("download_url", download_url), ("access_url", access_url)):
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                resource_type = build_resource_type(url, file_format)
                supported_intents = infer_supported_intents(content_scope, resource_type)
                download_priority = classify_download_priority(domain_relevance, resource_type)

                if resource_type in {"table_html_markdown", "graphic_html_markdown"}:
                    extension = ".md"
                else:
                    extension = Path(urlparse(url).path).suffix or ".bin"

                filename = f"{dataset_code}__{slugify(title)}{extension}"
                destination = SOURCE_ROOT / resource_type / filename

                record = {
                    "dataset_id": dataset_id,
                    "dataset_code": dataset_code,
                    "title": title,
                    "description": description,
                    "publisher_name": publisher_name,
                    "maintainer_name": maintainer_name,
                    "landing_page": landing_page,
                    "keywords": keywords,
                    "themes": themes,
                    "domain_relevance": domain_relevance,
                    "content_scope": content_scope,
                    "resource_type": resource_type,
                    "supported_intents": supported_intents,
                    "url_type": url_type,
                    "url": url,
                    "format": file_format,
                    "media_type": media_type,
                    "license": license_url,
                    "download_priority": download_priority,
                    "destination": str(destination),
                }
                manifest.append(record)
                resource_urls.append(record)

            resources.extend(resource_urls)

        normalized.append(
            {
                "dataset_id": dataset_id,
                "dataset_code": dataset_code,
                "title": title,
                "description": description,
                "keywords": keywords,
                "themes": themes,
                "publisher_name": publisher_name,
                "maintainer_name": maintainer_name,
                "landing_page": landing_page,
                "domain_relevance": domain_relevance,
                "content_scope": content_scope,
                "resources": resources,
            }
        )

    return normalized, manifest


def write_outputs(normalized: List[dict], manifest: List[dict]) -> None:
    DATASETS_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    LINKS_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no work]"
    completed = int(width * current / total)
    bar = "#" * completed + "-" * (width - completed)
    return f"[{bar}] {current}/{total}"


def download_from_manifest(manifest: List[dict], min_priority: str, limit: Optional[int]) -> None:
    selected = [item for item in manifest if should_download(item["download_priority"], min_priority)]
    if limit is not None:
        selected = selected[:limit]
    print(f"Downloading {len(selected)} selected resources into {SOURCE_ROOT} ...")

    total = len(selected)
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for index, item in enumerate(selected, start=1):
        destination = Path(item["destination"])
        progress = render_progress(index, total)
        short_name = destination.name
        print(f"{progress} {item['resource_type']} {short_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            skipped_count += 1
            print("  -> already exists, skipping")
            continue

        success, error = download_resource(item, destination)
        if success:
            downloaded_count += 1
            print("  -> downloaded")
        else:
            failed_count += 1
            print(f"Failed: {item['url']} -> {error}")

    print(
        f"Download summary: downloaded={downloaded_count}, skipped={skipped_count}, failed={failed_count}, total={total}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect, classify, and download BMFTR/GovData metadata resources for HoWie-RAG.")
    parser.add_argument("--use-local", action="store_true", help="Use the cached raw metadata catalog if available.")
    parser.add_argument("--download", action="store_true", help="Download classified resources into source_data/govdata.")
    parser.add_argument(
        "--min-priority",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum resource priority to download.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of resources to download.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    catalog = load_catalog(use_local=args.use_local)
    normalized, manifest = normalize_catalog(catalog)
    write_outputs(normalized, manifest)

    print(f"Datasets normalized: {len(normalized)}")
    print(f"Resource records written: {len(manifest)}")
    print(f"Enriched datasets written to: {DATASETS_PATH}")
    print(f"Download manifest written to: {MANIFEST_PATH}")

    if args.download:
        download_from_manifest(manifest, min_priority=args.min_priority, limit=args.limit)
        print(f"Downloads completed into: {SOURCE_ROOT}")


if __name__ == "__main__":
    main()
