from __future__ import annotations

import json
import re
from typing import List, Tuple

from config.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    NOVELTY_RATING_SCALE, FEASIBILITY_RATING_SCALE, EVALUATION_MODLE,
    LAMBDA_NOVELTY, LAMBDA_FEASIBILITY, TOKEN_LOG_PATH2)
from .models import Idea
from .models import PatentSemanticNetwork
from .llm import evaluator


def _extract_score(
    output: str, key: str, min_s: float = 1.0, max_s: float = 5.0
) -> float:
    """Parse numerical scores from the model output; return intermediate values when parsing fails."""
    try:
        return float(json.loads(output)[key])
    except Exception:
        pass

    text = output.strip()
    patterns = [
        rf"(?:score|rating|final score)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        rf"give it a\s*(\d+(?:\.\d+)?)",
        rf"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*{int(max_s)}",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if min_s <= v <= max_s:
                return v

    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if m:
        v = float(m.group(1))
        if min_s <= v <= max_s:
            return v

    return (min_s + max_s) / 2.0


def _norm(score: float, scale: float) -> float:
    return max(0.0, min(1.0, (score - 1.0) / (scale - 1.0)))




_NOVELTY_PROMPT = """\
    You are a patent examination expert, please evaluate the "novelty of the scientific idea" for the following ideas.
    You MUST output your result in **valid JSON format only**.
    Do NOT output any explanatory text, markdown, or comments outside the JSON.

    The JSON schema must be exactly as follows:
    {{
    "novelty_score": <decimal number between 1 and {scale}>
    }}

    [idea text]:
    {idea_text}

    Rating criteria:
    Novelty (1–{scale}): How original and innovative the idea is compared to existing research.
        - {scale}: Extremely novel and groundbreaking. The idea introduces new, unexplored concepts or radically shifts the direction of the field.
        - 4: Highly original. The idea is new and innovative but may still be building upon existing concepts or research.
        - 3: Moderately original. The idea brings some new insights but is similar to existing work or follows well-established concepts.
        - 2: Slightly original. The idea offers minor variations or incremental improvements to existing research but lacks substantial novelty.
        - 1: Not original. The idea closely resembles existing research with little to no innovation.
"""

_FEASIBILITY_PROMPT = """\
    You are a patent examination expert, please evaluate the "feasibility of the scientific idea" in terms of logic, consistency, and combinatorial reasonableness.
    You MUST output your result in **valid JSON format only**.
    Do NOT output any explanatory text, markdown, or comments outside the JSON.

    The JSON schema must be exactly as follows:
    {{
    "feasibility_score": <decimal number between 1 and {scale}>
    }}

    【Full Text of Technical Solution】:
    {idea_text}

    Scoring Criteria:
    Feasibility (1–{scale}): How realistic and practical the idea is to implement in current scientific and technological conditions.
    - {scale}: Fully feasible. The idea can be realistically executed with existing methods, data, and resources, and the plan for implementation is clear and practical.
    - 4: Highly feasible. The idea is feasible with current technologies but may require some advancements or additional resources.
    - 3: Moderately feasible. The idea faces significant practical challenges, requiring considerable advancements in technology or data.
    - 2: Slightly feasible. The idea is difficult to implement with current resources and would need significant breakthroughs.
    - 1: Not feasible. The idea is impractical and unlikely to be implemented with current technologies or methods. 
"""



class NewIdeaEvaluator:
    """Calculate Comprehensive Reward"""

    def __init__(self, target_patents: List[str] | None = None) -> None:
        self.target_patents = target_patents or []
        if PatentSemanticNetwork is not None:
            self._net = PatentSemanticNetwork(
                uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD,
            )
        else:
            self._net = None

    def close(self) -> None:
        if self._net is not None:
            try:
                self._net.close()
            except Exception:
                pass


    def _call(self, prompt: str) -> str:
        resp = evaluator(prompt, EVALUATION_MODLE, TOKEN_LOG_PATH2).strip()
        return resp

    def _novelty(self, idea: Idea) -> float:
        scale = NOVELTY_RATING_SCALE
        prompt = _NOVELTY_PROMPT.format(
            scale=scale, idea_text=idea.text
        )
        try:
            raw = _extract_score(self._call(prompt), "novelty_score", 1, scale)
            return _norm(raw, scale)
        except Exception as exc:
            print(f"[Evaluator] Novelty evaluation failed: {exc}")
            return 0.5

    def _feasibility(self, idea: Idea) -> float:
        scale = FEASIBILITY_RATING_SCALE
        prompt = _FEASIBILITY_PROMPT.format(
            scale=scale, idea_text=idea.text
        )
        try:
            raw = _extract_score(self._call(prompt), "feasibility_score", 1, scale)
            return _norm(raw, scale)
        except Exception as exc:
            print(f"[Evaluator] Feaseibility evaluation failed: {exc}")
            return 0.5



    def evaluate(self, idea: Idea) -> Tuple[float, float, float, float]:
        """
        Comprehensively evaluate the idea.

        Returns
        (overall, novelty, feasibility)
            All normalized to [0, 1]. Overall is the weighted average of the items.
        """
        novelty      = self._novelty(idea)
        feasibility  = self._feasibility(idea)
        overall      = (
            LAMBDA_NOVELTY     * novelty
            + LAMBDA_FEASIBILITY * feasibility
        )
        return overall, novelty, feasibility
