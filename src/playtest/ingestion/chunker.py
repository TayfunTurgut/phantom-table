import re
from typing import Any, cast

import chromadb
import tiktoken
from openai import OpenAI

from playtest.config import get_settings, maybe_wrap_openai

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _looks_like_header(block: str) -> bool:
    """A header is a short single line without sentence-ending punctuation."""
    stripped = block.strip()
    if "\n" in stripped:
        return False
    return len(stripped) <= 60 and not stripped.endswith((".", ":", "!", "?"))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _split_long_block(text: str, max_chunk_tokens: int) -> list[str]:
    """Split a too-long block at sentence boundaries, packing up to the token budget."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and _count_tokens(candidate) > max_chunk_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_rulebook(text: str, max_chunk_tokens: int = 500, game_name: str = "game") -> list[dict]:
    """Split rulebook into chunks. Each chunk is {"id": str, "text": str, "metadata": dict}.

    Splits on double-newline (paragraph boundaries), prepends the current section header as
    context, and further splits any block that exceeds max_chunk_tokens at sentence boundaries.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    chunks: list[dict] = []
    section = "Introduction"
    index = 0
    for block in blocks:
        if _looks_like_header(block):
            section = block.strip()
            continue

        pieces = (
            [block]
            if _count_tokens(block) <= max_chunk_tokens
            else _split_long_block(block, max_chunk_tokens)
        )
        for piece in pieces:
            chunks.append(
                {
                    "id": f"{game_name}_{index}",
                    "text": f"{section}\n\n{piece}",
                    "metadata": {"section": section, "chunk_index": index},
                }
            )
            index += 1

    return chunks


def _embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = maybe_wrap_openai(OpenAI(api_key=settings.openai_api_key))
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_and_store(chunks: list[dict], collection_name: str, persist_dir: str) -> None:
    """Embed chunks and store in a ChromaDB persistent collection."""
    embeddings = _embed_texts([chunk["text"] for chunk in chunks])

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    # upsert (not add) as a safety net: the pipeline always nukes the config dir
    # before this runs, so the collection is normally empty, but upsert keeps the
    # write idempotent even if it is ever called against a pre-existing collection.
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=cast(Any, embeddings),
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def query_collection(
    query: str, collection_name: str, persist_dir: str, n_results: int = 3
) -> list[str]:
    """Embed the query and return the top-n matching chunk texts from the collection."""
    query_embedding = _embed_texts([query])[0]

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    result = collection.query(query_embeddings=cast(Any, [query_embedding]), n_results=n_results)
    documents = result.get("documents") or [[]]
    return documents[0]
