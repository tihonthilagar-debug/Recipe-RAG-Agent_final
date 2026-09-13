import io
import os
import re
from pathlib import Path
from typing import List, Dict

import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import random
import time

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "recipe_documents"
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))


class RecipeRAG:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.chroma = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_builtin_documents()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return [e.values for e in result.embeddings]

    def _chunk(self, text: str, size: int = 900, overlap: int = 120) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def _index_builtin_documents(self):
        if self.collection.count() > 0:
            return
        docs = list(DATA_DIR.glob("*.md")) + list(DATA_DIR.glob("*.txt"))
        for path in docs:
            self._index_text(path.read_text(encoding="utf-8"), path.name)

    def _index_text(self, text: str, source: str) -> int:
        chunks = self._chunk(text)
        if not chunks:
            return 0
        embeddings = self._embed(chunks)
        ids = [f"{source}-{i}" for i in range(len(chunks))]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        return 1

    def add_uploaded_files(self, uploads) -> int:
        added = 0
        for upload in uploads:
            name = upload.name
            raw = upload.getvalue()
            if name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = raw.decode("utf-8", errors="ignore")
            if self._index_text(text, name):
                added += 1
        return added

    def retrieve(self, query: str, k: int = 5):
        q_embedding = self._embed([query])[0]
        result = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        return list(zip(docs, metas))

    def answer(self, query: str, cuisine: str, dietary_restrictions: List[str],
               max_time: int, available_ingredients: str) -> Dict:
        retrieval_query = (
            f"{query}. Cuisine: {cuisine}. Dietary restrictions: {', '.join(dietary_restrictions) or 'none'}. "
            f"Maximum cooking time: {max_time} minutes. Available ingredients: {available_ingredients or 'not specified'}."
        )
        retrieved = self.retrieve(retrieval_query, k=5)

        context = "\n\n".join(
            f"[Source: {meta.get('source', 'unknown')}]\n{doc}"
            for doc, meta in retrieved
        )

        system = """You are the Recipe Generator Agent in a Document Q&A RAG system.
Use the retrieved recipe context as the primary factual source. Do not invent a recipe
as if it came from the documents. You may adapt ingredients and instructions to satisfy
the user's constraints. Clearly mark any adaptation.

Return a polished answer with these sections:
1. Recipe name
2. Why it matches the user's request
3. Ingredients with quantities
4. Step-by-step cooking instructions
5. Substitutions
6. Approximate nutrition per serving (calories, protein, carbs, fat) and state that it is an estimate
7. Shopping list for missing ingredients
8. Short food-safety/allergen note

Be practical and concise. If the retrieved context is insufficient for a precise claim,
say so rather than fabricating it."""

        prompt = f"""USER REQUEST:
{query}

PERSONALIZATION:
Cuisine: {cuisine}
Dietary restrictions: {', '.join(dietary_restrictions) or 'None'}
Maximum cooking time: {max_time} minutes
Available ingredients: {available_ingredients or 'Not specified'}

RETRIEVED RECIPE KNOWLEDGE:
{context}
"""

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        # Gemini can temporarily return HTTP 503 when a model pool is under heavy load.
        # Retry with exponential backoff, then switch to a lighter fallback model.
        models_to_try = [GEN_MODEL]
        if FALLBACK_MODEL and FALLBACK_MODEL != GEN_MODEL:
            models_to_try.append(FALLBACK_MODEL)

        response = None
        last_error = None
        for model_name in models_to_try:
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            max_output_tokens=1800,
                        ),
                    )
                    break
                except genai_errors.ServerError as exc:
                    last_error = exc
                    if getattr(exc, "code", None) != 503:
                        raise
                    if attempt < MAX_RETRIES - 1:
                        delay = min(20, 2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(delay)
            if response is not None:
                break

        if response is None:
            raise RuntimeError(
                "Gemini is temporarily unavailable (503) on both the primary and fallback models. "
                "Please wait a moment and try again."
            ) from last_error

        sources = []
        for _, meta in retrieved:
            src = meta.get("source", "unknown")
            if src not in sources:
                sources.append(src)

        return {
            "answer": response.text,
            "sources": sources,
            "retrieved_chunks": len(retrieved),
        }
