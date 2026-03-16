import numpy as np
from typing import List
from .models import Patent, GeneratedIdea
from .utils import evaluator
from config.config import (
    LAMBDA_NOVELTY, LAMBDA_FEASIBILITY,
    NOVELTY_RATING_SCALE, FEASIBILITY_RATING_SCALE
)
import json
import re

def extract_score(model_output: str, min_score=0, max_score=5):
    """
    Extract a numeric score from English model output.
    Returns float or None if not found.
    """

    text = model_output.strip()

    
    semantic_patterns = [
        rf"(?:score|rating|rated|final score|final rating)\s*[:=]?\s*({min_score}(?:\.\d+)?|[2-4](?:\.\d+)?|{max_score}(?:\.0+)?)",
        rf"give it a\s*({min_score}(?:\.\d+)?|[2-4](?:\.\d+)?|{max_score}(?:\.0+)?)",
        rf"({min_score}(?:\.\d+)?|[2-4](?:\.\d+)?|{max_score}(?:\.0+)?)\s*(?:/|out of)\s*{max_score}"
    ]

    for pattern in semantic_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    fallback_pattern = rf"\b({min_score}(?:\.\d+)?|[2-4](?:\.\d+)?|{max_score}(?:\.0+)?)\b"
    match = re.search(fallback_pattern, text)
    if match:
        return float(match.group(1))

    return 1.0


def safe_parse(output, type_key):
    try:
        return json.loads(output)[type_key]
    except Exception:
        m = extract_score(output, 1, NOVELTY_RATING_SCALE if type_key == "novelty_score" else FEASIBILITY_RATING_SCALE)
        return m
    
class RewardCalculator:
    """
    Reward Calculator - Comprehensive Evaluation of Idea Quality

    Evaluation Dimensions:
    1. Novelty: Degree of innovation of features
    2. Feasibility: Possibility of implementing the technical solution
    """
    
    def __init__(self, target_main_patents: List[Patent], user_query: str):
        """
        Initialize Reward Calculator

        Args:
            target_main_patents: List of target main patents
            user_query: User query
        """
        self.target_main_patents = target_main_patents
        self.user_query = user_query
        self.lambda_novelty = LAMBDA_NOVELTY
        self.lambda_feasibility = LAMBDA_FEASIBILITY
    
    def calculate_reward(self, idea: GeneratedIdea) -> float:
        """
        Calculate Comprehensive Reward

        Args:
            idea: Generated idea

        Returns:
            Comprehensive reward score (0-1)
        """
        novelty_score = self._novelty_score(idea)
        feasibility_score = self._feasibility_score(idea)
        
        reward = (self.lambda_novelty * novelty_score +
                 self.lambda_feasibility * feasibility_score)
        print(f"Reward calculation details - Novelty: {novelty_score:.3f}, Feasibility: {feasibility_score:.3f} => Comprehensive Reward: {reward:.3f}")
        return reward
    
    def _semantic_similarity(self, idea: GeneratedIdea) -> float:
        """
        Calculate semantic similarity

        Compare the cosine similarity between the creative embedding and the target patent embedding

        Args:
            idea: Generated idea

        Returns:
            Similarity score (0-1)
        """
        if not self.target_main_patents:
            return 0.5
        
        similarities = []
        for main_patent in self.target_main_patents:
            cos_sim = np.dot(idea.embedding, main_patent.embedding)
            similarities.append(cos_sim)
        
        avg_similarity = np.mean(similarities)
        return max(0, min(1, avg_similarity))
    
    def _novelty_score(self, idea: GeneratedIdea) -> float:
        """
        Calculate Novelty Score

        Use GPT technology to evaluate the degree of innovation in the technical solution

        Args:
            idea: Generated creativity

        Returns:
            Standardized novelty score (0-1)
        """

        prompt = f"""
    You are a patent examination expert, please evaluate the "novelty of the scientific idea" for the following ideas.
    You MUST output your result in **valid JSON format only**.
    Do NOT output any explanatory text, markdown, or comments outside the JSON.

    The JSON schema must be exactly as follows:
    {{
    "novelty_score": <decimal number between 1 and {NOVELTY_RATING_SCALE}>
    }}

    [idea text]:
    {idea.text}

    Rating criteria:
    Novelty (1–{NOVELTY_RATING_SCALE}): How original and innovative the idea is compared to existing research.
        - {NOVELTY_RATING_SCALE}: Extremely novel and groundbreaking. The idea introduces new, unexplored concepts or radically shifts the direction of the field.
        - {NOVELTY_RATING_SCALE-1}: Highly original. The idea is new and innovative but may still be building upon existing concepts or research.
        - 3: Moderately original. The idea brings some new insights but is similar to existing work or follows well-established concepts.
        - 2: Slightly original. The idea offers minor variations or incremental improvements to existing research but lacks substantial novelty.
        - 1: Not original. The idea closely resembles existing research with little to no innovation.
    """

        try:
            resp = evaluator(prompt, "deepseek-v3.2").strip()
            print(f"\n Novelty Score Response: {resp}")
            score0 = safe_parse(resp, "novelty_score")
            # score0 = extract_score(resp, 1, NOVELTY_RATING_SCALE)
            score = float(score0)
            x = (score - 3) / 2
            return 1 / (1 + np.exp(-5 * x))
        except Exception:
            return 0.3
    
    def _feasibility_score(self, idea: GeneratedIdea) -> float:
        """
        Calculate Feasibility Score

        Evaluate the logical consistency and implementability of the technical solution

        Args:
            idea: Generated idea

        Returns:
            Standardized feasibility score (0-1)
        """
        prompt = f"""
    You are a patent examination expert, please evaluate the "feasibility of the scientific idea" in terms of logic, consistency, and combinatorial reasonableness.
    You MUST output your result in **valid JSON format only**.
    Do NOT output any explanatory text, markdown, or comments outside the JSON.

    The JSON schema must be exactly as follows:
    {{
    "feasibility_score": <decimal number between 1 and {FEASIBILITY_RATING_SCALE}>
    }}

    【Full Text of Technical Solution】:
    {idea.text}

    【List of Claims】:
    {idea.claims}

    Scoring Criteria:
    Feasibility (1–{FEASIBILITY_RATING_SCALE}): How realistic and practical the idea is to implement in current scientific and technological conditions.
    - {FEASIBILITY_RATING_SCALE}: Fully feasible. The idea can be realistically executed with existing methods, data, and resources, and the plan for implementation is clear and practical.
    - {FEASIBILITY_RATING_SCALE-1}: Highly feasible. The idea is feasible with current technologies but may require some advancements or additional resources.
    - 3: Moderately feasible. The idea faces significant practical challenges, requiring considerable advancements in technology or data.
    - 2: Slightly feasible. The idea is difficult to implement with current resources and would need significant breakthroughs.
    - 1: Not feasible. The idea is impractical and unlikely to be implemented with current technologies or methods. 
    """
        try:
            resp = evaluator(prompt, "deepseek-v3.2").strip()
            score0 = safe_parse(resp, "feasibility_score")
            # score0 = extract_score(resp, 1, FEASIBILITY_RATING_SCALE)
            score = float(score0)
  
            x = (score - 3) / 2
            return 1 / (1 + np.exp(-5 * x))
        except Exception:
            return 0.3

    def estimate_reward_from_ref_patents(self, ref_patents: List) -> float:
        """
        Rapid reward estimation for the current path based on the reference patent list (lightweight version).

        Purpose: Provide an estimate of path quality before generating a complete idea, for guiding probabilistic decision-making.

        Implementation (lightweight):
        - Use the mean of the reference patent embeddings as the path embedding
        - Use semantic similarity as the dominant factor, supplemented by a constant proportion of novelty/coherence.

        Return value: Estimated score within the range of 0-1.
        """
        try:
            if not ref_patents:
                return 0.0

            # Merge the embedding of the referenced patents as path representation
            embeddings = [rp.embedding for rp in ref_patents if hasattr(rp, 'embedding') and rp.embedding is not None]
            if not embeddings:
                return 0.0
            fused = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm

            # Use the semantic similarity of the path to the target patent as the dominant factor
            if not self.target_main_patents:
                semantic_score = 0.5
            else:
                sims = []
                for mp in self.target_main_patents:
                    if hasattr(mp, 'embedding') and mp.embedding is not None:
                        sims.append(float(np.dot(fused, mp.embedding)))
                semantic_score = float(np.mean(sims)) if sims else 0.5
                semantic_score = max(0.0, min(1.0, semantic_score))

            novelty_score = 0.5
            feasibility_score = 0.5

            reward = (self.lambda_novelty * novelty_score +
                      self.lambda_feasibility * feasibility_score)

            return float(max(0.0, min(1.0, reward)))
        except Exception:
            return 0.0
