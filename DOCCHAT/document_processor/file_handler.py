"""
Document Processor — parses PDF, DOCX, TXT, MD files into LangChain Document chunks.
Uses pypdf for PDFs, python-docx for DOCX, and plain read for TXT/MD.
"""
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docchat.config import constants
from docchat.config.settings import settings
from docchat.utils.logging import logger


class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = 0
        for f in files:
            fpath = f if isinstance(f, str) else f.name
            total_size += os.path.getsize(fpath)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit"
            )

    def process(self, files: List) -> List[Document]:
        """Process files with caching."""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                fpath = file if isinstance(file, str) else file.name
                with open(fpath, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {fpath}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {fpath}")
                    chunks = self._process_file(fpath)
                    self._save_to_cache(chunks, cache_path)

                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(
                        chunk.page_content.encode()
                    )
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {file}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, filepath: str) -> List[Document]:
        """Parse a file into LangChain Document chunks."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext not in constants.ALLOWED_TYPES:
            logger.warning(f"Skipping unsupported file type: {filepath}")
            return []

        text = ""

        if ext == ".pdf":
            text = self._read_pdf(filepath)
        elif ext == ".docx":
            text = self._read_docx(filepath)
        elif ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if not text.strip():
            logger.warning(f"No text extracted from {filepath}")
            return []

        # Split text into chunks
        chunks = self.splitter.create_documents(
            [text],
            metadatas=[{"source": os.path.basename(filepath)}],
        )
        # Assign stable chunk IDs for evaluation traceability
        source_name = os.path.basename(filepath)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{source_name}_{i:04d}"
        return chunks

    def _read_pdf(self, filepath: str) -> str:
        """Extract text from PDF using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(filepath)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n\n".join(pages)
        except Exception as e:
            logger.error(f"PDF extraction error for {filepath}: {e}")
            return ""

    def _read_docx(self, filepath: str) -> str:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document as DocxDocument

            doc = DocxDocument(filepath)
            return "\n\n".join(
                para.text for para in doc.paragraphs if para.text.strip()
            )
        except Exception as e:
            logger.error(f"DOCX extraction error for {filepath}: {e}")
            return ""

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List[Document], cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"timestamp": datetime.now().timestamp(), "chunks": chunks}, f
            )

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(
            cache_path.stat().st_mtime
        )
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
