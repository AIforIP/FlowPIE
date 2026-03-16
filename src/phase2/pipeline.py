from __future__ import annotations

import csv
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from config.config import (
    CONVERGENCE_THRESHOLD, MAX_GENERATION,
    MUTATION_PROBABILITY, COMBINATION, TOP_K,
)
from .evaluator import NewIdeaEvaluator
from .evolution import evolve_sample
from .llm import LLMInterface
from .models import Idea
from .retriever import PatentRetriever
from .selection import load_phase1_data

_CSV_FIELDS = [
    "sample_idx", "generation", "idea_index",
    "overall_score", "novelty_score", "feasibility_score",
    "generation_num", "is_mutated", "source_type",
]


def run_evolution_pipeline(
    json_path: str,
    output_dir: Optional[str] = None,
    max_generations: int = MAX_GENERATION,
    num_combinations: int = COMBINATION,
    top_k: int = TOP_K,
    mutation_probability: float = MUTATION_PROBABILITY,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    max_samples: Optional[int] = None,
) -> None:
    """
    json_path         : Phase1 output json path
    output_dir        : output path (By default, create evolved/ at the same level as json_path.)
    max_generations   : maximum number of generations for each sample
    num_combinations  : number of pairs to cross in each generation
    top_k             : population size
    mutation_probability : mutation probability
    convergence_threshold : early stopping threshold
    max_samples       : only process the first N samples (for debugging)
    """
    print("=== Phase2 Evolution idea generation process initiated. ===")
    print(f"Input: {json_path}")

    data = load_phase1_data(json_path)
    print(f"total samples: {len(data)}")
    if max_samples is not None:
        data = data[:max_samples]
        print(f"only processing the first {max_samples} samples.")

    # Initialize components
    llm        = LLMInterface()
    retriever  = PatentRetriever()
    evaluator  = NewIdeaEvaluator()

    # Output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_path), "evolved")
    os.makedirs(output_dir, exist_ok=True)

    ts           = int(time.time())
    csv_path     = os.path.join(output_dir, f"evolution_history_{ts}.csv")
    results_path = os.path.join(output_dir, f"evolved_ideas_{ts}.json")

    all_results: List[Dict[str, Any]] = []
    csvfile = None

    try:
        csvfile = open(csv_path, "w", newline="", encoding="utf-8")
        writer  = csv.DictWriter(csvfile, fieldnames=_CSV_FIELDS)
        writer.writeheader()

        for idx, sample in enumerate(data):
            print(f"\n>>> Sample {idx+1}/{len(data)} <<<")
            try:
                best = evolve_sample(
                    sample=sample,
                    llm=llm,
                    retriever=retriever,
                    evaluator=evaluator,
                    max_generations=max_generations,
                    num_combinations=num_combinations,
                    top_k=top_k,
                    mutation_probability=mutation_probability,
                    convergence_threshold=convergence_threshold,
                    csv_writer=writer,
                )
                result = _build_result(sample, best)
                all_results.append(result)
                print(f"Done, best score: {result['evolved_best_score']:.3f}")
            except Exception:
                print(f"Sample #{idx} processing failed:")
                traceback.print_exc()

            # Save results every time a sample is processed, in case of interruption
            _save_json(results_path, all_results, json_path, output_dir,
                       max_generations, num_combinations, top_k,
                       mutation_probability, convergence_threshold, ts)

    finally:
        retriever.close()
        evaluator.close()
        if csvfile:
            csvfile.close()

    # Summary statistics
    print(f"\n=== Process completed. ===")
    print(f"Results JSON: {results_path}")
    print(f"Processed total samples: {len(all_results)}")
    if all_results:
        orig    = [r["original_best_reward"] for r in all_results if r.get("original_best_reward")]
        evolved = [r["evolved_best_score"]   for r in all_results if r.get("evolved_best_score")]
        if orig:
            print(f"  initial ideas's best score: {sum(orig)/len(orig):.4f}")
        if evolved:
            print(f"  evoluated ideas's best score: {sum(evolved)/len(evolved):.4f}")



def _build_result(sample: Dict[str, Any], ideas: List[Idea]) -> Dict[str, Any]:
    return {
        "sample_idx":           sample.get("idx"),
        "topic":                sample.get("topic", ""),
        "target_paper":         sample.get("target_paper", ""),
        "original_best_idea":   sample.get("best_idea", ""),
        "original_best_reward": sample.get("best_reward", 0.0),
        "evolved_top_ideas": [
            {
                "text":           idea.text,
                "patent_path":    idea.patent_path,
                "overall_score":  idea.overall_score,
                "novelty_score":  idea.novelty_score,
                "feasibility_score": idea.feasibility_score,
                "generation":     idea.generation,
                "is_mutated":     idea.is_mutated,
                "metadata":       idea.metadata,
            }
            for idea in ideas
        ],
        "evolved_best_score": ideas[0].overall_score if ideas else 0.0,
    }


def _save_json(
    path: str,
    results: List[Dict[str, Any]],
    json_path: str,
    output_dir: str,
    max_generations: int,
    num_combinations: int,
    top_k: int,
    mutation_probability: float,
    convergence_threshold: float,
    timestamp: int,
) -> None:
    payload = {
        "meta": {
            "input_path":           json_path,
            "output_dir":           output_dir,
            "max_generations":      max_generations,
            "num_combinations":     num_combinations,
            "top_k":                top_k,
            "mutation_probability": mutation_probability,
            "convergence_threshold": convergence_threshold,
            "num_samples":          len(results),
            "timestamp":            timestamp,
        },
        "samples": results,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
