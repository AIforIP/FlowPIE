from __future__ import annotations

import copy
import random
import re
from typing import Dict, List, Optional

from .llm import LLMInterface
from .models import Idea
from .retriever import PatentRetriever



def parse_idea_text(idea_text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in idea_text.splitlines():
        m = re.match(r"\s*#*\s*\**\s*\(([A-E])\)", line)
        if m:
            current = m.group(1)
            sections.setdefault(current, [])
            sections[current].append(line)
        elif current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


_CROSSOVER_PROMPT = """\
    You are an expert innovation designer. Your task is to intelligently fuse two scientific ideas into one superior idea, leveraging the provided literature context including both relevant literature and diverse island literature.

    ## Research Topic:
    {topic}

    ## Isolation Island Literature Context:
    {patent_context}

    Scientific Idea A (score: {score_a:.3f}):
    {idea_a}

    Scientific Idea B (score: {score_b:.3f}):
    {idea_b}


    ## Requirement:
    1. Carefully analyze both ideas and identify their strengths
    2. Study the literature context to understand existing science and identify gaps
    3. Intelligently merge the best elements from both ideas
    4. Leverage insights from island literature to enhance novelty and diversity
    5. Create a NEW, COHERENT idea that combines their advantages
    6. Ensure all five parts (A-E) are present and well-integrated
    7. The new idea should be more innovative, feasible, and diverse than either parent

    ## Output Format:
        ### (A) Core Method Description
            (Your fused core method)
        ### (B) Functional Principle 
            (Your fused functional principles)
        ### (C) Concrete Workflow 
            (Your fused workflow)
        ### (D) Potential Innovation Directions 
            (Your fused innovation points)
        ### (E) Experimental Design
            (Your fused experimental design)
"""


def crossover(
    idea_a: Idea,
    idea_b: Idea,
    llm: LLMInterface,
    retriever: PatentRetriever,
    topic: str,
) -> Optional[Idea]:
    """
    Use LLM to cross-fuse two parent ideas into one child idea.
    """
    combined_path = list(set(idea_a.patent_path + idea_b.patent_path))
    patent_context = retriever.get_evolution_context(patent_path=combined_path)

    prompt = _CROSSOVER_PROMPT.format(
        topic=topic,
        patent_context=patent_context,
        score_a=idea_a.overall_score,
        idea_a=idea_a.text,
        score_b=idea_b.overall_score,
        idea_b=idea_b.text,
    )

    response = llm.call(prompt)
    if not response or not response.strip():
        return None

    parts = parse_idea_text(response)
    if not parts:
        return None

    return Idea(
        text=response.strip(),
        parts=parts,
        patent_path=combined_path,
        overall_score=0.0,   
        generation=max(idea_a.generation, idea_b.generation) + 1,
        metadata={
            "source": "crossover",
            "parents": [
                {"text": idea_a.text[:100] + "...", "score": idea_a.overall_score},
                {"text": idea_b.text[:100] + "...", "score": idea_b.overall_score},
            ],
        },
    )



_MUTATION_PROMPT = """\


## Original Idea (Score: {score:.3f})
{idea_text}

## Instructions
1. Identify weaknesses or areas for improvement.
2. Draw inspiration from the patent context.
3. Introduce novel technical elements while maintaining feasibility.
4. Strengthen all five parts (A–E).

## Output Format

### (A) Core Concept Description
[improved core concept]

### (B) Conceptual Functional Principle
[improved functional principles]

### (C) High-Level Conceptual Workflow
[improved workflow]

### (D) Potential Innovation Directions
[improved innovation points]

### (E) Theoretical Experimental Design
[improved experimental design]

    You are an expert innovation improver. Your task is to enhance and mutate an existing scientific idea to make it more novel and technically sound, leveraging diverse literature information.
    
    ## Research Topic:
    {topic}

    ## Isolation Island Literature Context:
    {patent_context}

    ## Original Idea (Score:{score:.3f}):
    {idea_text}

    ## Requirement:
    1. Identify potential weaknesses or areas for improvement
    2. Study the literature context to find inspiration from both relevant and island literature
    3. Introduce novel scientific elements or approaches inspired by diverse literature sources
    4.  Enhance the innovation level while maintaining feasibility
    5. Improve clarity and scientific depth
    6. Make sure all five parts (A-E) are strengthened
    7. Incorporate insights from island literature to increase diversity and novelty

    ## Output Format:
       ## (A) Core Method Description
            (Your improved core method)
       ## (B) Functional Principle
            (Your improved functional principles)
       ## (C) Concrete Workflow
            (Your improved workflow)
       ## (D) Potential Innovation Directions
            (Your improved innovation points)
       ## (E) Experimental Design
            (Your improved experimental design)
"""


def mutate(
    idea: Idea,
    llm: LLMInterface,
    retriever: PatentRetriever,
    topic: str,
    mutation_probability: float = 0.3,
) -> Idea:
    """
    Mutate the idea with a probability of mutation_probability. If the LLM call fails or the mutation is not triggered, return the original idea.
    """
    if random.random() > mutation_probability:
        return idea

    patent_context = retriever.get_evolution_context(patent_path=idea.patent_path)
    prompt = _MUTATION_PROMPT.format(
        topic=topic,
        patent_context=patent_context,
        score=idea.overall_score,
        idea_text=idea.text,
    )

    response = llm.call(prompt)
    if not response or not response.strip():
        return idea

    parts = parse_idea_text(response)
    if not parts:
        return idea

    return Idea(
        text=response.strip(),
        parts=parts,
        patent_path=copy.deepcopy(idea.patent_path),
        overall_score=0.0,   
        generation=idea.generation,
        is_mutated=True,
        metadata={
            "source": "mutation",
            "parent": {"text": idea.text[:100] + "...", "score": idea.overall_score},
        },
    )
