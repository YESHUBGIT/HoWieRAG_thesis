from pathlib import Path
from typing import List

from pypdf import PdfReader

from howie_rag.core.schemas import Document
from howie_rag.core.utils import stable_id
from howie_rag.document_classification import classify_document_content


_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _read_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def _read_supported_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return _read_pdf_text(file_path)
    return file_path.read_text(encoding="utf-8")


def load_text_documents(directory_path: str) -> List[Document]:
    directory = Path(directory_path)
    documents: List[Document] = []

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue

        text = _read_supported_text(file_path)
        classification = classify_document_content(
            title=file_path.stem,
            text=text,
            file_type=file_path.suffix.lower().lstrip("."),
        )
        documents.append(
            Document(
                doc_id=stable_id(f"{file_path.name}:{text}"),
                title=file_path.stem,
                text=text,
                metadata={
                    "source_path": str(file_path),
                    "file_type": file_path.suffix.lower().lstrip("."),
                    **classification,
                },
            )
        )

    return documents
