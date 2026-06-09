"""
agents/ingestion_agent.py
Handles PDF/URL ingestion:
  - Text extraction with PyPDF + pdfplumber
  - Semantic chunking (respects sentence boundaries)
  - Vision Agent: CLIP-based image/chart description
  - Stores chunks into the hybrid vector store
"""
import io
import uuid
import base64
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

import pypdf
import pdfplumber
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.config import settings
from core.models import DocumentChunk, IngestedDocument, DocumentSource
from core.vector_store import get_vector_store


# ─── Vision Agent (CLIP) ──────────────────────────────────────────────────────

class VisionAgent:
    """
    Uses OpenAI CLIP to describe images found in PDFs.
    Falls back to a simple caption if CLIP is unavailable.
    Your EDGE — leverages your production CLIP experience!
    """

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._available = False
        self._load_clip()

    def _load_clip(self):
        try:
            import clip
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)
            self._available = True
            logger.info(f"CLIP VisionAgent loaded on {self._device}")
        except Exception as e:
            logger.warning(f"CLIP not available, using OpenAI vision fallback: {e}")
            self._available = False

    def describe_image(self, image: Image.Image, context: str = "") -> str:
        """
        Describe an image using CLIP zero-shot classification
        with domain-relevant candidate labels.
        """
        if not self._available:
            return self._openai_vision_fallback(image, context)

        try:
            import clip
            import torch

            candidates = [
                "a bar chart showing data",
                "a line graph with trends",
                "a pie chart",
                "a table with rows and columns",
                "a flowchart or diagram",
                "a scanned text page",
                "a photograph",
                "a schematic or technical drawing",
                "a map or geographic visualization",
                "mathematical equations or formulas",
            ]

            image_input = self._preprocess(image).unsqueeze(0).to(self._device)
            text_tokens = clip.tokenize(candidates).to(self._device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_tokens)
                logits, _ = self._model(image_input, text_tokens)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            top_idx = probs.argmax()
            confidence = probs[top_idx]
            description = candidates[top_idx]

            return f"[Visual content: {description} (confidence: {confidence:.0%}). Context: {context[:100]}]"

        except Exception as e:
            logger.warning(f"CLIP inference failed: {e}")
            return f"[Visual content detected. Context: {context[:100]}]"

    def _openai_vision_fallback(self, image: Image.Image, context: str = "") -> str:
        """Use OpenAI GPT-4o-mini vision API as fallback for image description."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)  # vision only — Groq has no vision API yet

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": f"Describe this image briefly for a document retrieval system. Context: {context[:200]}",
                            },
                        ],
                    }
                ],
                max_tokens=150,
            )
            return f"[Visual: {response.choices[0].message.content}]"
        except Exception as e:
            logger.warning(f"OpenAI vision fallback failed: {e}")
            return "[Visual content detected — description unavailable]"


# ─── Text Chunker ─────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits text into overlapping chunks respecting sentence boundaries.
    Uses LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


# ─── Ingestion Agent ──────────────────────────────────────────────────────────

class IngestionAgent:
    """
    Main ingestion pipeline:
    1. Extract text + images from PDF
    2. Run VisionAgent on each image
    3. Chunk text with SemanticChunker
    4. Build DocumentChunk objects
    5. Store in HybridVectorStore
    """

    def __init__(self):
        self.vision_agent = VisionAgent()
        self.chunker = SemanticChunker()
        self.vector_store = get_vector_store()

    def ingest_pdf(self, pdf_path: str, doc_id: Optional[str] = None) -> IngestedDocument:
        """Parse a PDF file and ingest all its content."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = doc_id or str(uuid.uuid4())
        logger.info(f"Ingesting PDF: {path.name} (doc_id={doc_id})")

        # Extract text + image chunks
        text_chunks, image_chunks = self._extract_pdf(str(path), doc_id)

        all_chunks = text_chunks + image_chunks

        if all_chunks:
            self.vector_store.add_chunks(all_chunks)

        doc = IngestedDocument(
            doc_id=doc_id,
            source=DocumentSource.PDF,
            source_path=str(path),
            title=path.stem,
            num_chunks=len(all_chunks),
            has_images=len(image_chunks) > 0,
            status="ready",
        )
        logger.info(
            f"Ingested {len(text_chunks)} text + {len(image_chunks)} image chunks from {path.name}"
        )
        return doc

    def ingest_text(self, text: str, title: str = "Text Document", doc_id: Optional[str] = None) -> IngestedDocument:
        """Ingest raw text."""
        doc_id = doc_id or str(uuid.uuid4())
        raw_chunks = self.chunker.split(text)

        chunks = [
            DocumentChunk(
                doc_id=doc_id,
                content=chunk,
                chunk_index=i,
                metadata={"source_path": title, "title": title},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

        self.vector_store.add_chunks(chunks)

        return IngestedDocument(
            doc_id=doc_id,
            source=DocumentSource.TEXT,
            source_path=title,
            title=title,
            num_chunks=len(chunks),
            status="ready",
        )

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _extract_pdf(
        self, pdf_path: str, doc_id: str
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """Extract text and image chunks from a PDF."""
        text_chunks = []
        image_chunks = []
        full_text_pages = []

        # Phase 1: Extract text per page using pdfplumber (better layout)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        full_text_pages.append(
                            (page_num + 1, page_text)
                        )
        except Exception as e:
            logger.warning(f"pdfplumber failed, falling back to pypdf: {e}")
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        full_text_pages.append((i + 1, text))

        # Chunk extracted text
        chunk_idx = 0
        for page_num, page_text in full_text_pages:
            raw_chunks = self.chunker.split(page_text)
            for chunk_text in raw_chunks:
                if len(chunk_text.strip()) < 20:
                    continue
                text_chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        metadata={
                            "source_path": pdf_path,
                            "page": page_num,
                            "type": "text",
                        },
                    )
                )
                chunk_idx += 1

        # Phase 2: Extract images using pypdf
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                img_idx = 0
                for page_num, page in enumerate(reader.pages):
                    if "/XObject" not in (page.get("/Resources") or {}):
                        continue
                    xobjects = page["/Resources"]["/XObject"].get_object()
                    for name, obj in xobjects.items():
                        obj = obj.get_object()
                        if obj.get("/Subtype") == "/Image":
                            try:
                                image = self._extract_image_from_xobject(obj)
                                if image and image.width > 50 and image.height > 50:
                                    # Get surrounding text context
                                    context = full_text_pages[min(page_num, len(full_text_pages)-1)][1][:200] if full_text_pages else ""
                                    description = self.vision_agent.describe_image(image, context)

                                    image_chunks.append(
                                        DocumentChunk(
                                            doc_id=doc_id,
                                            content=description,
                                            chunk_index=chunk_idx + img_idx,
                                            has_image=True,
                                            image_description=description,
                                            metadata={
                                                "source_path": pdf_path,
                                                "page": page_num + 1,
                                                "type": "image",
                                                "image_index": img_idx,
                                            },
                                        )
                                    )
                                    img_idx += 1
                            except Exception as e:
                                logger.debug(f"Image extraction skipped: {e}")
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")

        return text_chunks, image_chunks

    def _extract_image_from_xobject(self, obj) -> Optional[Image.Image]:
        """Convert a PDF XObject to a PIL Image."""
        try:
            data = obj.get_data()
            filter_type = obj.get("/Filter")

            if filter_type == "/DCTDecode":
                return Image.open(io.BytesIO(data))
            elif filter_type == "/FlateDecode":
                width = int(obj["/Width"])
                height = int(obj["/Height"])
                color_space = obj.get("/ColorSpace", "/DeviceRGB")
                mode = "RGB" if "RGB" in str(color_space) else "L"
                return Image.frombytes(mode, (width, height), data)
            else:
                width = int(obj.get("/Width", 100))
                height = int(obj.get("/Height", 100))
                return Image.frombytes("RGB", (width, height), data)
        except Exception:
            return None
