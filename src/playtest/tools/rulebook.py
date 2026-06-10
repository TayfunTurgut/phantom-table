"""Rulebook query tool backed by the game's embedded ChromaDB collection."""

from pathlib import Path
from typing import Any, cast

import chromadb

from playtest.ingestion.chunker import _embed_texts


class RulebookTool:
    """Vector search over the rulebook chunks embedded during ingestion."""

    def __init__(self, game_config_dir: str, game_name: str) -> None:
        """Prepare the rulebook tool for a game.

        The ChromaDB collection is named by the config-directory basename (the slug
        used at ingestion), not the display ``game_name``. The collection is loaded
        lazily on the first query so constructing this tool stays cheap and offline.
        """
        self.game_name = game_name
        self._persist_dir = str(Path(game_config_dir) / "chromadb")
        self._collection_name = Path(game_config_dir).name
        self._collection: Any = None
        self._query_log: list[str] = []
        self._query_cache: dict[str, str] = {}

    def _get_collection(self) -> Any:
        # Assumption: the ChromaDB collection is not deleted (e.g. by re-ingestion)
        # after this tool is constructed. In the PoC, ingestion and gameplay never
        # overlap in the same process, so the lazily cached client/collection below
        # always points at a live path.
        if self._collection is None:
            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_collection(name=self._collection_name)
        return self._collection

    def query(self, query: str, n_results: int = 3) -> str:
        """Search the embedded rulebook and return the top-k chunks as text.

        Each chunk is prefixed with its section name (when available) for context.
        Results are cached per query string (``n_results`` never varies — the tool schema
        does not expose it), so only the first occurrence of a query pays for an OpenAI
        embedding request. Repeats still land in the query log for analytics.
        """
        self._query_log.append(query)
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        collection = self._get_collection()
        query_embedding = _embed_texts([query])[0]
        result = collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        blocks: list[str] = []
        for i, text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else None
            section = metadata.get("section") if isinstance(metadata, dict) else None
            header = f"[{section}]\n" if section else ""
            blocks.append(f"{header}{text}")

        result_text = "\n\n---\n\n".join(blocks)
        self._query_cache[query] = result_text
        return result_text

    def get_query_log(self) -> list[str]:
        """Return every rulebook query string made this game, in order."""
        return list(self._query_log)

    def as_openai_schema(self) -> dict:
        """Return the OpenAI function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": "query_rulebook",
                "description": (
                    "Search the game's rulebook for relevant rules and return the "
                    "most relevant passages. Use this when you are unsure how a rule "
                    "works or need to confirm the legality or effect of an action."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The rules question or topic to search for.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why you want to look this up in the rulebook.",
                        },
                    },
                    "required": ["query", "reasoning"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
