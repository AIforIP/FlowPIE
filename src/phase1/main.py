"""
main.py FlowPIE
"""

import os
import json
from tqdm import tqdm
from config.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    DEFAULT_TERMINATION_PROBABILITY, DEFAULT_TOP_K,
    MAX_ITERATIONS, DEFAULT_MAX_DEPTH,
    RESULTS_DIR, BENCHMARK_AGG_CSV, BENCHMARK_PER_QUERY_CSV,
    BENCHMARK_CURVE_PNG, BENCHMARK_COMP_PNG,
    PHASE1_JSON_PATH, PHASE1_OUTPUT_PATH,
    Swan_Project, Swan_workspace, Swan_experiment_name
)
from .mcts_flow import PatentInnovationMCTSFlow
from .utils import create_fulltext_index, save_results_to_json, load_test_data
import os
try:
    import swanlab
except Exception:
    swanlab = None
import numpy as np
import matplotlib.pyplot as plt


def main(data_path: str = "your_input_data.json",
         output_path: str = "your_output_data.json",
         p_term: float = DEFAULT_TERMINATION_PROBABILITY,
         top_k: int = DEFAULT_TOP_K,
         iterations: int = MAX_ITERATIONS,
         max_depth: int = DEFAULT_MAX_DEPTH,
         process_count: int = None,
         swan_mode: str = 'cloud',
         swan_project: str = Swan_Project,
         swan_workspace: str = Swan_workspace,
         swan_experiment_name: str = Swan_experiment_name,
         swan_logdir: str = None):
    """ 
    Args:
        data_path: input data path
        output_path: output data path
        p_term: termination probability
        top_k: top-k for the first layer
        iterations: number of iterations
        max_depth: maximum search depth
        process_count: number of data to process (None means all)
    """
    

    try:
        data = load_test_data(data_path)
        print(f"Load {len(data)} test data")
    except Exception as e:
        print(f"Error while loading test data: {e}")
        return

    run = None
    try:
        if swanlab is not None:
            mode = os.getenv('SWAN_MODE', swan_mode)
            project = os.getenv('SWAN_PROJECT', swan_project)
            workspace = os.getenv('SWAN_WORKSPACE', swan_workspace)
            exp_name = os.getenv('SWAN_EXPERIMENT_NAME', swan_experiment_name)
            logdir = os.getenv('SWAN_LOGDIR', swan_logdir)

            print(f"Initializing SwanLab run: mode={mode}, project={project}, workspace={workspace}, exp={exp_name}")
            try:
                run = swanlab.init(project=project, workspace=workspace, experiment_name=exp_name, mode=mode, logdir=logdir)
                print('SwanLab run initialized:', getattr(run, 'id', None))
            except Exception as e:
                print('SwanLab init failed:', e)
        else:
            print('swanlab package not available; skipping SwanLab init')
    except Exception as e:
        print('Error while initializing swanlab:', e)
    
  
    datalist = []

    benchmark_per_query_avg = []   
    benchmark_per_query_top5 = []  
    
    process_count = min(process_count, len(data))
    for i in tqdm(range(0, process_count)):
        user_query = data[i]['topic']
        idx = data[i]["index"]
        print(f"\n{'='*80}")
        print(f"Processing query {i+1}/{process_count}")
        print(f"Query: {user_query}")
        print('='*80)
        
        mcts = None
        
        try:
            if i == 0:  
                create_fulltext_index(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            
            mcts = PatentInnovationMCTSFlow(
                NEO4J_URI, 
                NEO4J_USER, 
                NEO4J_PASSWORD,
                user_query,
                p_term=p_term,
                top_k=top_k
            )
            
            print("\n" + "="*80)
            print("Flow-Guided MCTS Start...")
            print("="*80)
            best_path, best_reward, best_idea, all_ideas = mcts.run(
                iterations=iterations,
                max_depth=max_depth
            )
            
            
            if all_ideas:
                all_idea_texts = [idea_info['idea'].text for idea_info in all_ideas]
                
                all_idea_details = [
                    {
                        'text': idea_info['idea'].text,
                        'claims': idea_info['idea'].claims,
                        'patent_path': idea_info['patent_path'],
                        'flow': idea_info['flow'],
                        'visits': idea_info['visits']
                    }
                    for idea_info in all_ideas
                ]
                
                datalist.append({
                    # "abstract": data[i].get("abstract", ""),
                    # "claims": data[i].get("claims", []),
                    "idx": idx,
                    "target_paper": data[i].get("target_paper", " "),
                    "topic": user_query,
                    # 'paperId': data[i].get("paperId", ""),
                    'title': data[i].get("title", ""),
                    # "abstract": user_query,
                    # "target_paper": data[i].get("target_paper", " "),
                    "best_idea": best_idea.text if best_idea else "",
                    "best_patents": best_path,
                    "best_reward": best_reward,
                    "total_ideas_count": len(all_ideas),
                    "ideas": all_idea_texts,
                    "ideas_details": all_idea_details
                })
                
               
                save_results_to_json(datalist, output_path)
                print(f"\nresults saved to {output_path}")
                print(f"genetate {len(all_ideas)} ideas")
            
            try:
                if hasattr(mcts, 'iter_avg_rewards'):
                    benchmark_per_query_avg.append(list(mcts.iter_avg_rewards))
                else:
                    benchmark_per_query_avg.append([])

                if hasattr(mcts, 'iter_top5_rewards'):
                    benchmark_per_query_top5.append(list(mcts.iter_top5_rewards))
                else:
                    benchmark_per_query_top5.append([])
            except Exception:
                benchmark_per_query_avg.append([])
                benchmark_per_query_top5.append([])
        except Exception as e:
            print(f"\nError while processing query {i+1}/{process_count}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if mcts is not None:
                mcts.close()
        # break
        
    
    print("\n" + "="*80)
    print(f"total queries: {process_count}")
    print(f"generate: {len(datalist)} results")
    print(f"saved: {output_path}")
    print("="*80)
    try:
        out_dir = RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)

        num_queries = len(benchmark_per_query_avg)
        max_iters = iterations
        arr = np.full((num_queries, max_iters), np.nan, dtype=float)
        for qi, row in enumerate(benchmark_per_query_avg):
            for j, v in enumerate(row[:max_iters]):
                arr[qi, j] = float(v)

        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        counts = np.sum(~np.isnan(arr), axis=0)
        ci95 = 1.96 * stds / np.sqrt(np.where(counts > 0, counts, 1))

        csv_bench = os.path.join(out_dir, BENCHMARK_AGG_CSV)
        with open(csv_bench, 'w') as fh:
            fh.write('iteration,mean,std,count,ci95_lower,ci95_upper\n')
            for idx in range(max_iters):
                mu = float(means[idx]) if not np.isnan(means[idx]) else 0.0
                sd = float(stds[idx]) if not np.isnan(stds[idx]) else 0.0
                n = int(counts[idx])
                ci = float(ci95[idx]) if not np.isnan(ci95[idx]) else 0.0
                fh.write(f"{idx+1},{mu},{sd},{n},{mu-ci},{mu+ci}\n")
        print(f"Saved benchmark agg CSV: {csv_bench}")

        csv_per_query = os.path.join(out_dir, BENCHMARK_PER_QUERY_CSV)
        with open(csv_per_query, 'w') as fh:
            header = ['query_idx'] + [f'iter_{i+1}' for i in range(max_iters)]
            fh.write(','.join(header) + '\n')
            for qi in range(num_queries):
                row = [str(qi)] + [str(x) if not np.isnan(x) else '' for x in arr[qi]]
                fh.write(','.join(row) + '\n')
        print(f"Saved per-query matrix CSV: {csv_per_query}")

        try:
            x = list(range(1, max_iters + 1))
            plt.figure(figsize=(8, 4))
            plt.plot(x, means, '-o', label='Mean Top-5 Avg')
            plt.fill_between(x, means - stds, means + stds, color='C0', alpha=0.2, label='±1 STD')
            plt.fill_between(x, means - ci95, means + ci95, color='C0', alpha=0.15, label='95% CI')
            plt.xlabel('Iteration')
            plt.ylabel('Top-5 Avg Reward')
            plt.title('Benchmark-level Top-5 Avg Reward per Iteration')
            plt.legend()
            bench_png = os.path.join(out_dir, BENCHMARK_CURVE_PNG)
            plt.tight_layout()
            plt.savefig(bench_png)
            plt.close()
            print(f"Saved benchmark plot: {bench_png}")
        except Exception as e:
            print('Failed to plot benchmark curve:', e)

        # Iteration 1 vs Iteration max comparison（box/violin）
        try:
            iter1 = arr[:, 0][~np.isnan(arr[:, 0])]
            itern = arr[:, max_iters-1][~np.isnan(arr[:, max_iters-1])]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # Violin plot
            axes[0].violinplot([iter1, itern], showmeans=True)
            axes[0].set_xticks([1, 2])
            axes[0].set_xticklabels(['Iter 1', f'Iter {max_iters}'])
            axes[0].set_title('Violin: Iter1 vs IterN Top-5 Avg')
            # Boxplot
            axes[1].boxplot([iter1, itern])
            axes[1].set_xticks([1, 2])
            axes[1].set_xticklabels(['Iter 1', f'Iter {max_iters}'])
            axes[1].set_title('Boxplot: Iter1 vs IterN Top-5 Avg')
            comp_png = os.path.join(out_dir, BENCHMARK_COMP_PNG)
            plt.tight_layout()
            plt.savefig(comp_png)
            plt.close()
            print(f"Saved iteration comparison plot: {comp_png}")
        except Exception as e:
            print('Failed to plot iteration comparison:', e)

        try:
            if swanlab is not None and run is not None:
                try:
                    if hasattr(swanlab, 'upload'):
                        swanlab.upload(csv_bench)
                        swanlab.upload(csv_per_query)
                except Exception:
                    pass

                try:
                    if hasattr(swanlab, 'upload'):
                        swanlab.upload(bench_png)
                        swanlab.upload(comp_png)
                except Exception:
                    pass

                try:
                    for idx, mu in enumerate(means, start=1):
                        try:
                            swanlab.log({'benchmark_mean': float(mu), 'benchmark_std': float(stds[idx-1])}, step=idx)
                        except Exception:
                            try:
                                swanlab.log({'benchmark_mean': float(mu)}, step=idx)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            print('SwanLab upload/logging failed')

    except Exception as e:
        print('Benchmark aggregation failed:', e)

    try:
        if swanlab is not None and run is not None:
            try:
                swanlab.finish()
                print('SwanLab run finished')
            except Exception as e:
                print('SwanLab finish failed:', e)
    except Exception:
        pass


def main_single_query(user_query: str,
                     p_term: float = DEFAULT_TERMINATION_PROBABILITY,
                     top_k: int = DEFAULT_TOP_K,
                     iterations: int = MAX_ITERATIONS,
                     max_depth: int = DEFAULT_MAX_DEPTH):
    mcts = None
    
    try:
        create_fulltext_index(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        mcts = PatentInnovationMCTSFlow(
            NEO4J_URI, 
            NEO4J_USER, 
            NEO4J_PASSWORD,
            user_query,
            p_term=p_term,
            top_k=top_k
        )
        
        best_path, best_reward, best_idea, all_ideas = mcts.run(
            iterations=iterations,
            max_depth=max_depth
        )
        
        return best_path, best_reward, best_idea, all_ideas
        
    finally:
        if mcts is not None:
            mcts.close()


if __name__ == "__main__":
    main(
        data_path=PHASE1_JSON_PATH,
        output_path=PHASE1_OUTPUT_PATH,
        process_count=1,  # 0 indicates processing all data
    )
    

