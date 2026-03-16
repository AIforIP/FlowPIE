from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Dict, List

from .models import Idea, IdeaPair
from .operators import parse_idea_text


def load_phase1_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_ideas_from_sample(sample: Dict[str, Any]) -> List[Idea]:
    ideas: List[Idea] = []
    for idx, detail in enumerate(sample.get("ideas_details") or []):
        text = detail.get("idea", "") or ""
        ideas.append(Idea(
            text=text,
            parts=parse_idea_text(text),
            patent_path=detail.get("patent_path") or [],
            overall_score=float(detail.get("flow", 0.0)),
            generation=0,
            metadata={"source": "initial", "original_index": idx},
        ))
    return ideas


def generate_idea_pairs(ideas: List[Idea]) -> List[IdeaPair]:
    pairs: List[IdeaPair] = [
        IdeaPair(
            idea1=ideas[i],
            idea2=ideas[j],
            avg_score=(ideas[i].overall_score + ideas[j].overall_score) / 2.0,
            combination_index=k,
        )
        for k, (i, j) in enumerate(combinations(range(len(ideas)), 2))
    ]
    pairs.sort(key=lambda p: p.avg_score, reverse=True)
    return pairs


def select_top_ideas(ideas: List[Idea], top_k: int = 5) -> List[Idea]:
    return sorted(ideas, key=lambda x: x.overall_score, reverse=True)[:top_k]
