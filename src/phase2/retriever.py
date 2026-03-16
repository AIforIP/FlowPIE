from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    USE_ISLAND_PATENTS, ISLAND_SIZE,
)
from .models import PatentSemanticNetwork




class PatentRetriever:
    def __init__(self) -> None:
        self._net = PatentSemanticNetwork(
            uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD,
        )
        self._cache: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        try:
            self._net.close()
        except Exception:
            pass

    # ── 单条专利 ──────────────────────────────────────────────────────────────

    def get_patent_details(self, patent_id: str) -> Optional[Dict[str, Any]]:
        if not patent_id:
            return None
        if patent_id in self._cache:
            return self._cache[patent_id]
        try:
            mp = self._net.get_main_patent(patent_id)
            if mp:
                info: Dict[str, Any] = {
                    "id":       mp.id,
                    "title":    mp.title or "",
                    "abstract": mp.abstract or "",
                    "claims":   mp.claims[:10] if mp.claims else [],
                }
                self._cache[patent_id] = info
                return info
        except Exception as exc:
            print(f"[PatentRetriever] Neo4j Search Error {patent_id}: {exc}")
        return None


    def get_patents_context(self, patent_ids: List[str]) -> str:
        snippets: List[str] = []
        for pid in patent_ids:
            info = self.get_patent_details(pid)
            if info:
                claims_str = "; ".join(info["claims"][:5]) if info["claims"] else "N/A"
                snippets.append(
                    f"[Patent: {info['id']}]\n"
                    f"Title: {info['title']}\n"
                    f"Abstract: {info['abstract']}\n"
                    f"Key Claims: {claims_str}"
                )
        return "\n\n".join(snippets)



    def get_island_patents(
        self, exclude_ids: List[str], limit: int = ISLAND_SIZE
    ) -> List[str]:
        """Randomly sample patent IDs not in the exclude_ids."""
        exclude_ids = [p for p in (exclude_ids or []) if p]
        try:
            with self._net.driver.session() as session:
                result = session.run(
                    """
                    MATCH (mp:MainPatent)
                    WHERE NOT mp.id IN $exclude_ids
                    RETURN mp.id AS id
                    ORDER BY rand()
                    LIMIT $limit
                    """,
                    exclude_ids=exclude_ids,
                    limit=limit,
                )
                return [r["id"] for r in result if r.get("id")]
        except Exception as exc:
            print(f"[PatentRetriever] Island Patent Sampling Error: {exc}")
            return []


    def get_evolution_context(
        self,
        patent_path: List[str],
        use_island: bool = USE_ISLAND_PATENTS,
        island_size: int = ISLAND_SIZE,
    ) -> str:
        """
        Build the context for crossover / mutation operators.
        If use_island=True, append random patents to enhance diversity.
        """

        context = self.get_patents_context(patent_path)
        if use_island:
            island_ids = self.get_island_patents(patent_path, limit=island_size)
            island_ctx = self.get_patents_context(island_ids)
            if island_ctx:
                context += "\n\n## Island Patents (for diversity enhancement)\n" + island_ctx
        return context
