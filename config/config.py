# ==================== Neo4j configuration ====================
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "xxxxxx"

# ==================== Swanlab configuration=======================
Swan_Project = "xxx"
Swan_workspace = "xxx"
Swan_experiment_name = "xxx"

# ==================== API configuration ====================
OPENAI_API_KEY = "xxx"
OPENAI_BASE_URL = "xxx"
OPENAI_MODEL = "xxx"
EVALUATION_MODLE = "xxx"
TEMPERATURE = 0.2

# ==================== Embedding configuration ====================
EMBEDDING_MODEL = "xxx"
EMBEDDING_DEVICE = "xxx"

# ==================== MCTS-Flow configuration ====================
DEFAULT_TERMINATION_PROBABILITY = 0.4   # default termination probability
DEFAULT_TOP_K = 3                       # first layer Top-K
MAX_ITERATIONS = 10                     # maximum iterations
DEFAULT_MAX_DEPTH = 6                   # default maximum depth

# ==================== MCTS extension probability configuration ====================
EXPAND_PATENT_PROB = 0.8               # probability of expanding to a patent node
OPTIMIZE_IDEA_PROB = 0.2               # probability of optimizing an existing idea
SIMULATION_CROSSOVER_PROB = 0.7        # probability of using adjacent patents for crossover in the simulation stage

# ==================== Reward calculation weight ====================
LAMBDA_NOVELTY = 0.5       # novelty weight
LAMBDA_FEASIBILITY = 0.5     # feasibility weight

# ====================== Idea generation configuration ====================
CLAIM_CROSSOVER_SAMPLE_SIZE = 12       # number of claims to sample for crossover
MUTATION_RATE = 0.5                    # semantic mutation rate
EXTRA_FEATURES_SAMPLE_SIZE = 10        # number of extra features to sample
NOVELTY_RATING_SCALE = 5               # novelty rating scale (1-5)
FEASIBILITY_RATING_SCALE = 5           # feasibility rating scale (1-5)

# ==================== Search configuration ====================
HYBRID_SEARCH_LIMIT = 20               # hybrid search maximum number 
HYBRID_SEARCH_ALPHA = 0.6              # hybrid search weight
RELATED_PATENTS_LIMIT = 3              # related patents search limit

# ==================== Flow / PUCT parameters ====================
CPUCT = 1.0                             # PUCT exploration coefficient
ALPHA_FLOW = 0.2                        # Pflow update learning rate (alpha)
GAMMA_DECAY = 0.99                      # reward time decay factor gamma

# ==================== Keywords extraction configuration ====================
KEYWORDS_COUNT_MIN = 5                  # minimum number of keywords
KEYWORDS_COUNT_MAX = 10                 # maximum number of keywords

# ==================== Other configuration ====================
PRINT_TREE_MAX_DEPTH = 3               # maximum depth of tree structure to print
TOP_IDEAS_DISPLAY_COUNT = 10           # number of top ideas to display

# ==================== Phase2 evolution configuration ====================
MUTATION_PROBABILITY = 0.3          # mutation probability
CONVERGENCE_THRESHOLD = 0.005       # convergence threshold
MAX_GENERATION = 10                 # maximum evolution generations
COMBINATION = 5                     # number of crossover pairs per generation
TOP_K = 5                           # number of top ideas to retain
MAX_SAMPLES = 1                     # only process the first N samples
USE_ISLAND_PATENTS = True           # whether to use island patents
ISLAND_SIZE = 5                     # island size

# ==================== Output/path configuration ====================
RESULTS_DIR = "flowpie/results/AIBench"
REWARD_RAW_CSV = "reward_iteration_raw.csv"
REWARD_TOP5_CSV = "reward_iteration_top5.csv"
BENCHMARK_AGG_CSV = "benchmark_top5_avg_agg.csv"
BENCHMARK_PER_QUERY_CSV = "benchmark_per_query_top5_avg.csv"
BENCHMARK_CURVE_PNG = "benchmark_top5_avg_curve.png"
BENCHMARK_COMP_PNG = "benchmark_iter1_vs_iterN.png"

# ==================== Single sequence plotting file names ====================
AVG_REWARD_PNG = "avg_reward_curve.png"
TOP_REWARD_PNG_PREFIX = "top{n}_reward_curve.png"  # format with n=1..5

# ==================== Token counting record configuration ====================
TOKEN_LOG_ENABLED = True
TOKEN_LOG_PATH1 = "flowpie/results/AIBench/token_usage_log1.jsonl"
TOKEN_LOG_PATH2 = "flowpie/results/AIBench/evolved/token_usage_log2.jsonl"

# ==================== Input/output configuration ====================
PHASE1_JSON_PATH = "flowpie/data/bench_topics_test.json"
PHASE1_OUTPUT_PATH = "flowpie/results/AIBench/AIBench_flow.json"
PHASE2_JSON_PATH = "flowpie/results/AIBench/AIBench_flow.json"
PHASE2_OUTPUT_DIR = "flowpie/results/AIBench/evolved"