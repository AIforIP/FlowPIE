import argparse
import sys
from config.config import (MAX_GENERATION, COMBINATION, TOP_K, MAX_SAMPLES,
                           MUTATION_PROBABILITY, CONVERGENCE_THRESHOLD,
                           PHASE2_JSON_PATH, PHASE2_OUTPUT_DIR)
from .pipeline import run_evolution_pipeline



def main(argv=None) -> None:    
    run_evolution_pipeline(
        json_path=PHASE2_JSON_PATH,
        output_dir=PHASE2_OUTPUT_DIR,
        max_generations=MAX_GENERATION,
        num_combinations=COMBINATION,
        top_k=TOP_K,
        mutation_probability=MUTATION_PROBABILITY,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        max_samples=MAX_SAMPLES,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
