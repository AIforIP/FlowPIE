from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

from config.config import (
    CONVERGENCE_THRESHOLD, MAX_GENERATION,
    MUTATION_PROBABILITY, COMBINATION, TOP_K,
)
from .evaluator import NewIdeaEvaluator
from .llm import LLMInterface
from .models import Idea
from .operators import crossover, mutate
from .retriever import PatentRetriever
from .selection import extract_ideas_from_sample, generate_idea_pairs, select_top_ideas


def evolve_sample(
    sample: Dict[str, Any],
    llm: LLMInterface,
    retriever: PatentRetriever,
    evaluator: NewIdeaEvaluator,
    max_generations: int = MAX_GENERATION,
    num_combinations: int = COMBINATION,
    top_k: int = TOP_K,
    mutation_probability: float = MUTATION_PROBABILITY,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    csv_writer: Optional[csv.DictWriter] = None,
) -> List[Idea]:
    """
    Run the evolutionary loop for a single Phase 1 sample.

    Returns:

    The top_k Ideas after the evolution ends.
    """
    topic      = sample.get("topic", "Unknown topic")
    sample_idx = sample.get("idx", 0)


    # ========== initial ideas ============================
    current_ideas = extract_ideas_from_sample(sample)
    print(f"  Initial ideas: {len(current_ideas)}")
    for i, idea in enumerate(current_ideas):
        print(f"    Idea {i+1}: score={idea.overall_score:.3f}")

    prev_top_mean: Optional[float] = None

    # =========== evolution loop ======================
    for gen in range(1, max_generations + 1):
        print(f"\n--- Generation {gen}/{max_generations} ---")


        pairs = generate_idea_pairs(current_ideas)
        if not pairs:
            print("Warning: idea is insufficient to generate combinations, terminate.")
            break

        selected = pairs[:num_combinations]
        print(f"  selecte {num_combinations} pairs to crossover: ")
        for i, p in enumerate(selected):
            i1 = current_ideas.index(p.idea1) + 1
            i2 = current_ideas.index(p.idea2) + 1
            print(f"    {i+1}. Idea{i1} + Idea{i2}  avg={p.avg_score:.3f}")

       
        children: List[Idea] = []
        for i, p in enumerate(selected):
            print(f"  crossover {i+1}...", end=" ")
            child = crossover(p.idea1, p.idea2, llm, retriever, topic)
            if child:
                children.append(child)
                print("✓")
            else:
                print("✗ failed")
        print(f"  generated ideas: {len(children)} through crossover")


        mutated: List[Idea] = [
            mutate(c, llm, retriever, topic, mutation_probability)
            for c in children
        ]


        for i, idea in enumerate(mutated):
            try:
                print(f"  evaluate idea {i+1}...", end=" ")
                overall, novelty, feasibility = evaluator.evaluate(idea)
                idea.overall_score      = overall
                idea.novelty_score   = novelty
                idea.feasibility_score = feasibility
                print(f"overall={overall:.3f}  novelty={novelty:.3f}"
                      f"  feasibility={feasibility:.3f}")
            except Exception as exc:
                print(f"evaluation failed: {exc}")
                idea.overall_score = idea.novelty_score = \
                    idea.feasibility_score = 0.5


        pool = current_ideas + mutated
        current_ideas = select_top_ideas(pool, top_k)
        print(f"  next generation Top-{top_k}:")
        for i, idea in enumerate(current_ideas):
            src = idea.metadata.get("source", "unknown")
            print(f"    {i+1}. [{src}]  score={idea.overall_score:.3f}")
            if csv_writer:
                csv_writer.writerow(_csv_row(sample_idx, gen, i, idea))


        if len(current_ideas) >= top_k:
            curr_mean = sum(x.overall_score for x in current_ideas[:top_k]) / top_k
            print(f"  Top-{top_k} mean: {curr_mean:.3f}")
            if prev_top_mean is not None:
                delta = abs(curr_mean - prev_top_mean)
                print(f"  Δ={delta:.4f}  Threshold={convergence_threshold}")
                if delta < convergence_threshold:
                    print("  converged, early terminate.")
                    break
            prev_top_mean = curr_mean

    print(f"\nEvolution complete, final Top-{top_k}:")
    for i, idea in enumerate(current_ideas):
        print(f"  {i+1}. [{idea.metadata.get('source','unknown')}]  "
              f"score={idea.overall_score:.3f}")

    return current_ideas



def _csv_row(
    sample_idx: int, generation: int, idea_index: int, idea: Idea
) -> Dict[str, Any]:
    return {
        "sample_idx":    sample_idx,
        "generation":    generation,
        "idea_index":    idea_index,
        "overall_score":    round(idea.overall_score,      4),
        "novelty_score": round(idea.novelty_score,   4),
        "feasibility_score": round(idea.feasibility_score, 4),
        "generation_num": idea.generation,
        "is_mutated":    idea.is_mutated,
        "source_type":   idea.metadata.get("source", "unknown"),
    }
