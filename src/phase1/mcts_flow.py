import random
import numpy as np
import uuid
from typing import List, Tuple, Optional

from pyparsing import line
from .models import PatentSemanticNetwork, ReferencePatent, GeneratedIdea

from .idea_generator import IdeaGenerator
from .reward import RewardCalculator
from .utils import extract_keywords_from_query
from .embedding import HFEmbeddingEncoder
from config.config import (
    DEFAULT_TERMINATION_PROBABILITY, DEFAULT_TOP_K,
    PRINT_TREE_MAX_DEPTH, TOP_IDEAS_DISPLAY_COUNT,
    RESULTS_DIR, REWARD_RAW_CSV, REWARD_TOP5_CSV, AVG_REWARD_PNG, TOP_REWARD_PNG_PREFIX,
    MUTATION_RATE, RELATED_PATENTS_LIMIT,
    EXPAND_PATENT_PROB, OPTIMIZE_IDEA_PROB, SIMULATION_CROSSOVER_PROB,
    CPUCT, ALPHA_FLOW, GAMMA_DECAY
)

from .flow_scoring import ucb_puct_score, pflow_update, normalize_pflows, decayed_reward

import os
import math

import swanlab

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

sw = None


class MCTSFlowNode:
    """
    Support for MCTS nodes with Flow DAG

    Features:
    - Supports both idea nodes and patent nodes
    - Uses Flow instead of the traditional UCB for node selection
    - Maintains edge-level flow and access statistics
    """

    def __init__(self, ref_patent_ids: List[str], parent=None, action=None,
                 is_idea_node=False, idea=None):
        """
        Initialize MCTS Flow Node

        Args:
            ref_patent_ids: List of patent IDs
            parent: Parent node
            action: Action leading to this node (patent ID)
            is_idea_node: Whether it is an idea node
            idea: Idea object (if it is an idea node)
        """
        self.ref_patent_ids = ref_patent_ids
        self.parent = parent
        self.action = action
        self.children = []

        self.is_idea_node = is_idea_node
        self.idea = idea

        # Flow statistics
        self.F = 0.0              # Total flow through this node
        self.N = 0                # Access count
        self.edge_F = {}          # Edge flow: {child_node: flow_value}
        self.edge_N = {}          # Edge visit counts: {child_node: visit_count}
        self.edge_R_sum = {}      # Edge cumulative reward
        self.P_flow = {}          # Edge prior P_flow (normalized)

    def add_child(self, action, is_idea_node=False, idea=None) -> 'MCTSFlowNode':
        """
        Add Sub-node
                Args:
                    action: Action (Patent ID)
                    is_idea_node: Whether it is an idea node
                    idea: Idea object

                Returns:
                    Newly created sub-node
        """
        if is_idea_node:
            child = MCTSFlowNode(
                self.ref_patent_ids.copy(),
                parent=self,
                action=action,
                is_idea_node=True,
                idea=idea
            )
        else:
            child = MCTSFlowNode(
                self.ref_patent_ids + [action],
                parent=self,
                action=action,
                is_idea_node=False
            )
        self.children.append(child)

        # Initialize the flow of edges and the number of accesses (the flow is redistributed by the parent node after expansion)
        self.edge_F[child] = 0.0
        self.edge_N[child] = 0
        # Edge-level cumulative reward and prior
        self.edge_R_sum[child] = 0.0
        self.P_flow[child] = 0.0

        return child

    def ucb_flow_score(self, child: 'MCTSFlowNode', c: float = 1.414) -> float:
        """
        Calculate UCB Score Based on Flow

        Args:
            child: child node
            c: exploration coefficient

        Returns:
            UCB score
        """
        # Delegate to centralized PUCT implementation which uses edge_R_sum, P_flow, edge_N and parent.N
        return ucb_puct_score(self, child, c_puct=c)

    def best_child(self, c: float = 1.414) -> Optional['MCTSFlowNode']:
        """
        Select the optimal child node (based on Flow UCB)

        Args:
            c: Exploration coefficient

        Returns:
            Optimal child node, returns None if there are no child nodes
        """
     
        patent_children = [
            child for child in self.children if not child.is_idea_node]
        if not patent_children:
            return None

        scores = [(child, self.ucb_flow_score(child, c))
                  for child in patent_children]
        max_score = max(score for _, score in scores)
        # Select a candidate with the highest score randomly to avoid always choosing the first one.
        best_candidates = [child for child,
                           score in scores if score == max_score]
        if len(best_candidates) == 1:
            return best_candidates[0]
        return random.choice(best_candidates)

    def redistribute_flow(self):
        """
        Redistribute the flow of the current node to its patent children's edges.
        The idea child node does not participate in the flow distribution.
        """ 

        patent_children = [
            child for child in self.children if not child.is_idea_node]
        if not patent_children:
            return

        # If the parent node has an explicit P_flow, it will be allocated according to the P_flow; otherwise, it will be allocated evenly.
        total_p = sum(self.P_flow.get(child, 0.0) for child in patent_children)
        if total_p <= 0:
            per = self.F / len(patent_children)
            for child in patent_children:
                self.P_flow[child] = 1.0 / len(patent_children)
                self.edge_F[child] = per
                child.F = per
        else:
            for child in patent_children:
                p = self.P_flow.get(child, 0.0) / (total_p + 1e-12)
                self.edge_F[child] = self.F * p
                child.F = self.edge_F[child]


class PatentInnovationMCTSFlow:
    """
    MCTS with Flow DAG Patent Innovation System

    Main Functions:
    - Mixed Search Initialization
    - Selection: Flow-based UCB Node Selection
    - Expansion: Graph Structure-driven Node Expansion
    - Idea Generation: Creative Generation and Evaluation
    - Flow Backup: Reverse Propagation of Flow Values
    """

    def __init__(self, uri: str, user: str, password: str, user_query: str,
                 p_term: float = DEFAULT_TERMINATION_PROBABILITY,
                 top_k: int = DEFAULT_TOP_K):
        """
        Initialize MCTS-Flow system

        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
            user_query: User query text
            p_term: Termination probability (0-1)
            top_k: Top-K number for the first layer expansion
        """
        self.network = PatentSemanticNetwork(uri, user, password)
        self.user_query = user_query
        self.p_term = p_term
        self.top_k = top_k
        self.keywords = []
        self.query_embedding = []
        self.encoder = None
        self.initial_ref_patents = []
        self.target_main_patents = []
        self.reward_calculator = None
        # track per-iteration average rewards for visualization (and raw lists)
        self.iter_avg_rewards = []
        self.iter_rewards_raw = []

        self._initialize()

    def _initialize(self):
        """Initialize the system - keyword extraction, embedding, initial search, etc."""
        print("="*80)
        print("Initializing MCTS-Flow DAG System")
        print("="*80)
        print(f"User Query: {self.user_query}")
        print(f"Termination Probability P_term: {self.p_term}")
        print(f"First Layer Top-K: {self.top_k}")

        self.keywords = extract_keywords_from_query(self.user_query)

        self.encoder = HFEmbeddingEncoder()
        self.query_embedding = self.encoder.encode(self.user_query)

        search_results = self.network.hybrid_search_reference_patents(
            query_text=self.user_query,
            keywords=self.keywords,
            query_embedding=self.query_embedding,
            limit=20,
            alpha=0.6
        )

        print(f"Retrieved {len(search_results)} patents")

        print("\nLoad patent information...")
        for result in search_results:
            ref_id = result['id']
            ref_patent = self.network.get_reference_patent(ref_id)

            if ref_patent:
                self.initial_ref_patents.append(ref_patent)

                if ref_patent.main_patent_id:
                    main_patent = self.network.get_main_patent(
                        ref_patent.main_patent_id)
                    if main_patent and main_patent not in self.target_main_patents:
                        self.target_main_patents.append(main_patent)

        # print(f"Retrieved {len(self.initial_ref_patents)} ReferencePatents")
        # print(f"Corresponding to {len(self.target_main_patents)} MainPatents")

        self.reward_calculator = RewardCalculator(
            self.target_main_patents, self.user_query)
        # track per-iteration top-5 rewards (for swanlab / plotting)
        self.iter_top5_rewards = []

        print("\nInitialization complete!")

    def print_tree_structure(self, node: MCTSFlowNode, prefix="", is_last=True,
                             depth=0, max_depth=PRINT_TREE_MAX_DEPTH, line_callback=None):
        """
        Print tree structure for debugging

        Args:
            node: Current node
            prefix: Prefix string
            is_last: Whether it is the last child node
            depth: Current depth
            max_depth: Maximum print depth
        """
        if depth > max_depth:
            return

        connector = "└── " if is_last else "├── "

        if node.parent is None:
            print("ROOT (N={}, F={:.3f})".format(node.N, node.F))
        else:
            node_type = "💡IDEA" if node.is_idea_node else "📄Patent"
            action_display = node.action if len(
                str(node.action)) < 30 else str(node.action)[:27] + "..."
            edge_flow = 0.0
            if node.parent:
                edge_flow = node.parent.edge_F.get(node, 0.0)

            print("{}{}[{}] {} (N={}, F={:.3f}, EdgeF={:.3f})".format(
                prefix, connector, node_type, action_display,
                node.N, node.F, edge_flow
            ))

        if line_callback:
            line_callback(line)

        if node.children:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension if node.parent is not None else ""

            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self.print_tree_structure(
                    child, new_prefix, is_last_child, depth + 1, max_depth, line_callback=line_callback)

    def selection(self, node: MCTSFlowNode, max_depth: int,
                  depth: int = 0, is_first_layer: bool = False) -> Tuple[MCTSFlowNode, List[MCTSFlowNode], bool]:
        """
        Step 1: Selection - Flow-based UCB Path Selection

        Starting from the root node, select the path to reach the leaf node using the UCB strategy

        Args:
            node: Current node
            max_depth: Maximum depth limit
            depth: Current depth

        Returns:
            (selected_node, trajectory, should_terminate)
        """
        print("\n【Selection - Flow UCB】")
        trajectory = [node]
        current = node
        should_terminate = False

        while depth < max_depth:
            # If the current node is the root node, then randomly select its leaf node.
            if is_first_layer:
                print(f"  Selecting from top-{self.top_k} patents")
                for i, ref_patent in enumerate(self.initial_ref_patents[:self.top_k], 1):
                    child = node.add_child(ref_patent.id)
                    print(f"  {i}. Adding patent node: {ref_patent.id}")
                current = current.best_child(c=1.414)
                print(current.ref_patent_ids)
                if current is None:
                    break
                trajectory.append(current)
                depth += 1

                node_type = "Idea" if current.is_idea_node else "Patent"
                ucb_score = current.parent.ucb_flow_score(
                    current) if current.parent else 0
                print(f"  选择 {node_type} 节点, UCB={ucb_score:.3f}, "
                      f"N={current.N}, F={current.F:.3f}")
                break

            if not current.children:
                break

            # If the current node has only one child and that child is an idea node, consider the branch to have converged and stop selecting.
            idea_children = [c for c in current.children if c.is_idea_node]
            if len(current.children) == 1 and len(idea_children) == 1:
                print("  Current node has only one idea child, stopping selection to optimize/expand")
                break


            current = current.best_child(c=1.414)
            print(current.ref_patent_ids)
            if current is None:
                break
            trajectory.append(current)
            depth += 1

            node_type = "Idea" if current.is_idea_node else "Patent"
            ucb_score = current.parent.ucb_flow_score(
                current) if current.parent else 0
            print(f"  Selecting {node_type} node, UCB={ucb_score:.3f}, "
                  f"N={current.N}, F={current.F:.3f}")

        print(f"  Reached node: {current.ref_patent_ids}, Depth={depth}")
        return current, trajectory, should_terminate

    def expansion(self, node: MCTSFlowNode, is_first_layer: bool = False) -> MCTSFlowNode:
        """
        Step 2: Expansion - Node Expansion

        Args:
            node: The node to be expanded
            is_first_layer: Whether it is the first layer (using top-K)

        Returns:
            The expanded node (could be the original node or a new child node)
        """
        print("\n【Expansion - Node Expansion】")

        if node.is_idea_node:
            print("  idea node, not expanding")
            return node

        # First Layer: Query-Guided Top-K Expansion
        if is_first_layer:
            print(f"  First Layer Expansion: Using top-{self.top_k} patents")
            for i, ref_patent in enumerate(self.initial_ref_patents[:self.top_k], 1):
                child = node.add_child(ref_patent.id)
                print(f"  {i}. Adding patent node: {ref_patent.id}")
   
            node.redistribute_flow()
            return node

        # Non-First Layer: Graph-Structure-Based Expansion or Idea Optimization (Based on Probability)
        if len(node.ref_patent_ids) == 0:
            print("  Root node, skipping expansion")
            return node

        last_ref_id = node.ref_patent_ids[-1]
        exclude_ids = set(node.ref_patent_ids)

        related_refs = self.network.get_related_reference_patents(
            last_ref_id,
            limit=RELATED_PATENTS_LIMIT,
            exclude_ids=exclude_ids
        )

        if not related_refs:
            print("  No expandable nodes")
            return node

        # Based on probability, decide whether to expand the patent node or optimize the existing idea.
        choice = random.random()
        print(f"  expansion choice prob={choice:.3f}")

        if choice < EXPAND_PATENT_PROB:
            print(
                f"  With probability {EXPAND_PATENT_PROB}, expanding patent neighbors: {len(related_refs)} candidates")
            for ref_id, _ in related_refs:
                child = node.add_child(ref_id)
                print(f"  Adding patent node: {ref_id}")
            # After expansion, redistribute flow from parent node to new child nodes
            node.redistribute_flow()
            return node

        # Optimize the branches of the existing idea nodes
        if choice < EXPAND_PATENT_PROB + OPTIMIZE_IDEA_PROB:
            print(f"  With probability {OPTIMIZE_IDEA_PROB}, triggering idea optimization")
            # If the current node itself is an idea node, optimize it directly.
            if node.is_idea_node and node.idea:
                self.optimize_idea_at_node(node)
                return node

            # Otherwise, prioritize optimizing its existing idea child nodes
            idea_children = [
                c for c in node.children if c.is_idea_node and c.idea]
            if idea_children:
                target = random.choice(idea_children)
                self.optimize_idea_at_node(target)
                return node

            for ref_id, _ in related_refs:
                child = node.add_child(ref_id)
                print(f"  Adding patent node: {ref_id}")
            node.redistribute_flow()
            return node

        for ref_id, _ in related_refs:
            child = node.add_child(ref_id)
            print(f"  Adding patent node (default): {ref_id}")
        node.redistribute_flow()

        return node

    def optimize_idea_at_node(self, node: MCTSFlowNode) -> float:
        """
        Optimize the specified idea node (semantic variation + regenerate text) and return the new reward.
        """
        print("\n【Idea Optimization】")

        # Find the reference patent chain corresponding to the said idea.
        ref_patents = [self.network.get_reference_patent(
            rid) for rid in node.ref_patent_ids]
        ref_patents = [rp for rp in ref_patents if rp]

        if not node.is_idea_node or not node.idea:
            print("  Target node is not an idea node or has no idea, skip optimization")
            return 0.0

        original_claims = node.idea.claims
        mutated_claims = IdeaGenerator.semantic_mutation(
            original_claims, ref_patents, mutation_rate=MUTATION_RATE)

        try:
            new_text = IdeaGenerator._generate_text_from_claims(
                mutated_claims, mutated_claims, ref_patents, self.user_query)
        except Exception:
            new_text = node.idea.text

        node.idea.claims = mutated_claims
        node.idea.text = new_text

        new_reward = self.reward_calculator.calculate_reward(node.idea)
        node.F = new_reward
        print(f"  Optimization completed, new Reward={new_reward:.3f}")
        return new_reward

    def dynamic_termination_probability(self, node: MCTSFlowNode, min_p: float = 0.02, max_p: float = 0.95) -> float:
        """
        Dynamic termination probability: Adjust the probability of generating an idea based on the flow/reward of the current node.
        Basic idea: If the flow/reward of the node is relatively high, increase the probability of termination (generating an idea).
        The return value is between [min_p, max_p].
        """
        try:
      
            ref_patents = [self.network.get_reference_patent(
                rid) for rid in node.ref_patent_ids]
            ref_patents = [rp for rp in ref_patents if rp]

            if ref_patents and self.reward_calculator is not None:
                est_reward = self.reward_calculator.estimate_reward_from_ref_patents(
                    ref_patents)
    
                p = min_p + (max_p - min_p) * \
                    float(max(0.0, min(1.0, est_reward)))
                return float(max(min_p, min(max_p, p)))

            parent_F = node.parent.F if node.parent is not None and node.parent.F > 0 else node.F
            ratio = 0.0
            if parent_F > 0:
                ratio = float(node.F) / (parent_F + 1e-12)
            p = min_p + (max_p - min_p) * \
                (1.0 / (1.0 + math.exp(-8.0 * (ratio - 0.15))))
            return float(max(min_p, min(max_p, p)))
        except Exception:
            return min(max_p, max(min_p, 0.1))

    def generate_idea_at_node(self, node: MCTSFlowNode, prev_node: Optional[MCTSFlowNode] = None) -> Tuple[float, Optional[MCTSFlowNode]]:
        """
        Step 3: Generate idea at the node and calculate the reward

        Args:
            node: The node for generating the idea

        Returns:
            The calculated reward value
        """
        print("\n【Idea Generation】")

        if not node.ref_patent_ids:
            print("  Root node, cannot generate idea")
            return 0.1, None

        # If a preceding node is provided and the probability allows, it is preferred to use crossover/mutation between the current node and the previous patent node.
        use_adjacent_crossover = False
        if prev_node and random.random() < SIMULATION_CROSSOVER_PROB:
            # Ensure that there is at least one patent ID on both sides.
            if prev_node.ref_patent_ids and node.ref_patent_ids:
                use_adjacent_crossover = True

        if use_adjacent_crossover:
            last_prev = prev_node.ref_patent_ids[-1]
            last_curr = node.ref_patent_ids[-1]
            ref_patents = [self.network.get_reference_patent(last_prev),
                           self.network.get_reference_patent(last_curr)]
            ref_patents = [rp for rp in ref_patents if rp]
            print(f"  Using adjacent patent crossover for simulation: {last_prev} <-> {last_curr}")
        else:
            ref_patents = [self.network.get_reference_patent(rid)
                           for rid in node.ref_patent_ids]
        ref_patents = [rp for rp in ref_patents if rp]

        if not ref_patents:
            print("  No valid patents, returning low reward")
            return 0.1, None

        idea = IdeaGenerator.generate_idea(ref_patents, self.user_query)

        # Add the idea as a child node under the current patent node, rather than converting the current patent node into an idea node.
        idea_action = f"idea_{uuid.uuid4().hex[:8]}"
        idea_child = node.add_child(
            action=idea_action, is_idea_node=True, idea=idea)

        reward = self.reward_calculator.calculate_reward(idea)
        idea_child.F = reward

        print(f"  Idea generation completed")
        print(f"  Number of claims: {len(idea.claims)}")
        print(f"  Reward: {reward:.3f}")

        return reward, idea_child

    def flow_backup(self, trajectory: List[MCTSFlowNode]):
        """
        Step 4: Flow Backup - Backward propagation of flow values

        Propagate flow from the end of the trajectory to the root node

        Args:
            trajectory: The node trajectory from root to leaf
        """
        print("\n【Flow Backup】")

        if not trajectory:
            return

        # Update the access count of the edges (each edge from the root to the leaf is updated only once), the flow of edge_F is allocated by the parent node during expansion.
        for i in range(len(trajectory) - 1):
            parent = trajectory[i]
            child = trajectory[i + 1]

            parent.edge_N[child] = parent.edge_N.get(child, 0) + 1

            if child.is_idea_node:

                r = child.F
                parent.edge_R_sum[child] = parent.edge_R_sum.get(
                    child, 0.0) + r

                # Calculated the reward_tilde with time decay: Here, the exact t, T cannot be obtained, so the no-decay fallback is used.
                r_tilde = r
                # update P_flow (EMA)
                old_p = parent.P_flow.get(child, 0.0)
                parent.P_flow[child] = pflow_update(
                    old_p, r_tilde, alpha=ALPHA_FLOW)

            print(
                f"  transport edge: ({parent.ref_patent_ids}) -> ({child.ref_patent_ids}) number={parent.edge_N[child]}")

        for node in trajectory:
            node.N += 1

        # Recalculate the total flow of nodes with child nodes
        # After updating edge_R_sum / P_flow for all edges, normalize the P_flow of each parent node with child nodes and redistribute edge_F.
        for node in trajectory:
            if node.children:
                # Only consider the patent sub-nodes for P_flow normalization
                patent_children = [
                    ch for ch in node.children if not ch.is_idea_node]
                if patent_children:
                    # ensure all patent_children have a P_flow entry
                    for ch in patent_children:
                        node.P_flow.setdefault(ch, 0.0)

                    node.P_flow = normalize_pflows(
                        {ch: node.P_flow[ch] for ch in patent_children})

                    # edge_F 与 child.F
                    for ch in patent_children:
                        node.edge_F[ch] = node.F * node.P_flow.get(ch, 0.0)
                        ch.F = node.edge_F[ch]

                # Recalculate node F as the sum of its outgoing edges
                node.F = sum(node.edge_F.get(child, 0.0)
                             for child in node.children)
                print(f"  Update Node F({node.ref_patent_ids}) = {node.F:.3f}")

    def collect_all_ideas(self, node: MCTSFlowNode, ideas_list: list):
        """
        Recursively collect all idea nodes

        Args:
            node: Current node
            ideas_list: List used to store ideas
        """
        if node.is_idea_node and node.idea:
            ideas_list.append({
                'idea': node.idea,
                'patent_path': node.ref_patent_ids.copy(),
                'flow': node.F,
                'visits': node.N
            })

        for child in node.children:
            self.collect_all_ideas(child, ideas_list)

    def run(self, iterations: int = 50, max_depth: int = 6) -> Tuple[list, float, GeneratedIdea, list]:
        """
        Execute a complete MCTS-Flow search

        Args:
            iterations: Number of iterations
            max_depth: Maximum search depth

        Returns:
            (best_path, best_reward, best_idea, all_ideas)
        """
        print("\n" + "="*80)
        print(f"Query: {self.user_query}")
        print(f"Iterations: {iterations}, Max Depth: {max_depth}")
        print(f"Termination Probability: {self.p_term}")
        print("="*80)

        root = MCTSFlowNode([])
        root.F = 100.0
        best_reward = 0.0
        best_idea = None
        best_path = None

        for iteration in range(iterations):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print('='*80)

            is_first_iter = (iteration == 0)
            # collect rewards of ideas generated in this iteration
            iteration_rewards = []
            leaf, trajectory, should_terminate = self.selection(
                root, max_depth, depth=0, is_first_layer=is_first_iter)

            if not should_terminate and len(leaf.ref_patent_ids) < max_depth and not leaf.is_idea_node:
                if not leaf.children:
                    is_first_layer = (len(leaf.ref_patent_ids)
                                      == 0 and is_first_iter)
                    leaf = self.expansion(leaf, is_first_layer=is_first_layer)

                if leaf.children:

                    dyn_p = self.dynamic_termination_probability(leaf)
                    print(f"  Dynamic Termination Probability={dyn_p:.3f} (Based on Node F={leaf.F:.3f})")
                    if random.random() < dyn_p:
                        print(f"  Triggered termination after expansion, generating idea at current node")
                        should_terminate = True
                    else:
                        selected_child = random.choice(leaf.children)
                        leaf = selected_child
                        trajectory.append(leaf)
                        print(f"  From the newly expanded child nodes: {leaf.action}")

            if not leaf.is_idea_node:
                prev_node = None
                for n in reversed(trajectory[:-1]):
                    if not n.is_idea_node and n.ref_patent_ids:
                        prev_node = n
                        break

                reward, idea_node = self.generate_idea_at_node(
                    leaf, prev_node=prev_node)
                if idea_node is not None:
                    trajectory.append(idea_node)
                    iteration_rewards.append(float(reward))
            else:
                print("  Already selected existing idea node, executing optimization based on current path and re-evaluating reward")
                reward = self.optimize_idea_at_node(leaf)
                iteration_rewards.append(float(reward))

            self.flow_backup(trajectory)

            try:
                self.iter_rewards_raw.append(iteration_rewards)
            except Exception:
                self.iter_rewards_raw.append([])

            if reward > best_reward:
                best_reward = reward
                # When generating new ideas, best_idea should come from the idea node; otherwise, use leaf.
                if not leaf.is_idea_node and 'idea_node' in locals() and idea_node is not None:
                    best_idea = idea_node.idea
                    best_path = idea_node.ref_patent_ids.copy()
                else:
                    best_idea = leaf.idea
                    best_path = leaf.ref_patent_ids.copy()
                print(f"\nFound a better idea! Reward={best_reward:.3f} ***")

            if (iteration + 1) % 1 == 0 or iteration == iterations - 1:
                self.print_tree_structure(root, max_depth=10, line_callback=getattr(self, '_tree_line_callback', None))

            # Collect the Top-5 idea rewards from this iteration (for subsequent drawing and reporting), and use the average of Top-K as this round's reward.
            try:
                current_ideas = []
                self.collect_all_ideas(root, current_ideas)
                current_ideas.sort(key=lambda x: x['flow'], reverse=True)
                flows = [float(x['flow']) for x in current_ideas]

                # top-k average (k = min(5, available))
                if flows:
                    k = min(5, len(flows))
                    topk = flows[:k]
                    avg_topk = float(np.mean(topk))
                else:
                    k = 0
                    topk = []
                    avg_topk = 0.0

                # pad top5 for storage
                top5 = topk + [0.0] * (5 - len(topk))
                self.iter_top5_rewards.append(top5)
                # 使用Top-k平均作为本轮avg
                self.iter_avg_rewards.append(avg_topk)
                print(f"  This iteration's Top-{k} Rewards: {topk}, AvgTopK={avg_topk:.3f}")

                # Attempt to report to swanlab (if available)
                try:
                    if swanlab is not None and hasattr(swanlab, 'log'):
                        log_dict = {'avg_reward': avg_topk}
                        for idx in range(5):
                            log_dict[f'top{idx+1}'] = top5[idx]
                  
                        try:
                            swanlab.log(log_dict, step=iteration + 1)
                        except TypeError:
                            for k_name, v in log_dict.items():
                                try:
                                    swanlab.log(
                                        {k_name: v}, step=iteration + 1)
                                except Exception:
                                    pass
                except Exception:
                    pass

            except Exception:
                self.iter_top5_rewards.append([0.0, 0.0, 0.0, 0.0, 0.0])
            try:
                if len(self.iter_avg_rewards) >= 8:
                    recent = self.iter_avg_rewards[-3:]
                    rmax = max(recent)
                    rmin = min(recent)
                    if (rmax - rmin) <= 0.002:
                        # print(f"\nEarly Stop Trigger: In the last 3 Top-5 Avg range={rmax - rmin:.3f} <= 0.002, terminate the search prematurely.")
                        break
            except Exception:
                pass
                self.iter_avg_rewards.append(0.0)


        all_ideas = []
        self.collect_all_ideas(root, all_ideas)
        print(f"Generated {len(all_ideas)} ideas")

        all_ideas.sort(key=lambda x: x['flow'], reverse=True)
        all_ideas = all_ideas[:5]

        self._print_final_results(
            root, best_path, best_reward, best_idea, all_ideas)

        try:
            out_dir = RESULTS_DIR
            os.makedirs(out_dir, exist_ok=True)
            # save raw per-iteration rewards as CSV for inspection
            try:
                csv_path = os.path.join(out_dir, REWARD_RAW_CSV)
                with open(csv_path, 'w') as fh:
                    fh.write('iteration,avg_reward,raw_rewards\n')
                    for i, raw in enumerate(self.iter_rewards_raw, 1):
                        avg = self.iter_avg_rewards[i-1] if i - \
                            1 < len(self.iter_avg_rewards) else 0.0
                        fh.write(f"{i},{avg},{raw}\n")
                print(f"Saved per-iteration raw rewards to: {csv_path}")
            except Exception:
                pass
            try:
                csv_top5 = os.path.join(out_dir, REWARD_TOP5_CSV)
                with open(csv_top5, 'w') as fh2:
                    fh2.write('iteration,top1,top2,top3,top4,top5\n')
                    for i, top5 in enumerate(self.iter_top5_rewards, 1):
                        fh2.write(','.join([str(i)] + [str(x)
                                  for x in top5]) + '\n')
                print(f"Saved per-iteration top5 rewards to: {csv_top5}")
            except Exception:
                csv_top5 = None

            try:
                if plt is not None:
                    # avg_reward
                    try:
                        if self.iter_avg_rewards:
                            avg_png = os.path.join(out_dir, AVG_REWARD_PNG)
                            plt.figure(figsize=(8, 4))
                            plt.plot(range(1, len(self.iter_avg_rewards) + 1),
                                     self.iter_avg_rewards, '-o', color='C1')
                            plt.xlabel('Iteration')
                            plt.ylabel('Avg Reward')
                            plt.title('Avg Reward per Iteration')
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(avg_png)
                            plt.close()
                            print(f"Saved avg reward plot: {avg_png}")
                    except Exception:
                        pass

                    # top1..top5
                    for k in range(5):
                        try:
                            yvals = [row[k] if k < len(
                                row) else 0.0 for row in self.iter_top5_rewards]
                            if any(v != 0.0 for v in yvals):
                                top_png = os.path.join(
                                    out_dir, TOP_REWARD_PNG_PREFIX.format(n=k+1))
                                plt.figure(figsize=(8, 4))
                                plt.plot(range(1, len(yvals) + 1), yvals,
                                         '-o', label=f'Top-{k+1}', color=f'C{k}')
                                plt.xlabel('Iteration')
                                plt.ylabel('Reward')
                                plt.title(f'Top-{k+1} Reward per Iteration')
                                plt.grid(True)
                                plt.tight_layout()
                                plt.savefig(top_png)
                                plt.close()
                                print(
                                    f"Saved Top-{k+1} reward plot: {top_png}")
                        except Exception:
                            pass
            except Exception as e:
                print('Failed to save per-series plots:', e)


            if swanlab is not None:
                try:
                    x = list(range(1, len(self.iter_avg_rewards) + 1))
                    series = {}
                    for k in range(5):
                        series[f'Top-{k+1}'] = [row[k] if k <
                                                len(row) else 0.0 for row in self.iter_top5_rewards]
                    series['Avg'] = self.iter_avg_rewards

        
                    if hasattr(swanlab, 'plot') and hasattr(swanlab.plot, 'line'):
                        try:
                            swanlab.plot.line(
                                x=x, y=series, title='Top-5 Rewards per Iteration', xlabel='Iteration', ylabel='Reward')
                            print(
                                'Plotted Top-5 reward curves using swanlab.plot.line')
                        except Exception:
                            for name, yvals in series.items():
                                try:
                                    swanlab.plot.line(
                                        x=x, y=yvals, title=f'{name} per Iteration', xlabel='Iteration', ylabel='Reward')
                                except Exception:
                                    pass
                    elif hasattr(swanlab, 'plot') and hasattr(swanlab.plot, 'Plot'):
                        try:
                            p = swanlab.plot.Plot()
                            for name, yvals in series.items():
                                try:
                                    p.line(x, yvals, label=name)
                                except Exception:
                                    pass
                            p.title('Top-5 Rewards per Iteration')
                            p.render()
                            print(
                                'Plotted Top-5 reward curves using swanlab.plot.Plot')
                        except Exception:
                            pass

                    try:
                        if csv_path and hasattr(swanlab, 'upload'):
                            swanlab.upload(csv_path)
                        if csv_top5 and hasattr(swanlab, 'upload'):
                            swanlab.upload(csv_top5)
                        # 上传单序列PNG
                        try:
                            avg_path = os.path.join(out_dir, AVG_REWARD_PNG)
                            if os.path.exists(avg_path) and hasattr(swanlab, 'upload'):
                                swanlab.upload(avg_path)
                        except Exception:
                            pass
                        try:
                            for k in range(5):
                                top_path = os.path.join(
                                    out_dir, TOP_REWARD_PNG_PREFIX.format(n=k+1))
                                if os.path.exists(top_path) and hasattr(swanlab, 'upload'):
                                    swanlab.upload(top_path)
                        except Exception:
                            pass
                    except Exception:
                        pass

                except Exception as e:
                    print(f'swanlab plotting/reporting attempt failed: {e}, reverted to matplotlib')
                    if plt is not None:
                        out_path = os.path.join(
                            out_dir, f'reward_iteration_plot.png')
                        plt.figure(figsize=(10, 5))
                        for k in range(5):
                            yvals = [row[k] if k < len(
                                row) else 0.0 for row in self.iter_top5_rewards]
                            plt.plot(range(1, len(yvals) + 1), yvals,
                                     marker='o', label=f'Top-{k+1}')
                        if self.iter_avg_rewards:
                            plt.plot(range(1, len(self.iter_avg_rewards) + 1),
                                     self.iter_avg_rewards, marker='x', linestyle='--', label='Avg')
                        plt.xlabel('Iteration')
                        plt.ylabel('Reward')
                        plt.title('Top-5 Rewards per Iteration')
                        plt.grid(True)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(out_path)
                        plt.close()
                        print(f"Top-5 Reward saved: {out_path}")
                    else:
                        print('matplotlib Not installed, unable to save rollback image.')
            elif plt is not None and self.iter_avg_rewards:
                out_path = os.path.join(out_dir, f'reward_iteration_plot.png')
                plt.figure(figsize=(8, 4))
                plt.plot(range(1, len(self.iter_avg_rewards) + 1),
                         self.iter_avg_rewards, '-o')
                plt.xlabel('Iteration')
                plt.ylabel('Avg Reward')
                plt.title('Avg Reward per Iteration')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
                print(f"Reward iteration curve saved: {out_path}")
            else:
                print('No visualization tool available or no data to plot, skipping plotting.')
        except Exception as e:
            print(f"Error occurred while plotting: {e}")

        return best_path, best_reward, best_idea, all_ideas

    def _print_final_results(self, root: MCTSFlowNode, best_path: list,
                             best_reward: float, best_idea: GeneratedIdea, all_ideas: list):
        print("\n" + "="*80)
        print("Flow-Guided MCTS complete!")
        print("="*80)
        print(f"  - Query: {self.user_query}")
        print(f"  - Generated ideas: {len(all_ideas)}")
        print(f"  - Best iterature path: {best_path}")
        print(f"  - Best reward: {best_reward:.3f}")
        print(f"  - Root node visits: {root.N}")
        print(f"  - Root node flow: {root.F:.3f}")

        if all_ideas:
            display_count = min(TOP_IDEAS_DISPLAY_COUNT, len(all_ideas))
            print("\n" + "="*80)
            print(f"Top-{display_count} Ideas (sorted by flow):")
            print("="*80)
            for i, idea_info in enumerate(all_ideas[:display_count], 1):
                print(f"\n{i}. Idea #{i}")
                print(f"   Patent path: {idea_info['patent_path']}")
                print(f"   Flow: {idea_info['flow']:.3f}")
                print(f"   Visits: {idea_info['visits']}")

        if best_path:
            print("\n" + "="*80)
            print("Best Innovation Path Details:")
            print("="*80)
            for i, ref_id in enumerate(best_path, 1):
                ref_patent = self.network.get_reference_patent(ref_id)
                if ref_patent:
                    print(f"\n{i}. ReferencePatent: {ref_id}")
                    print(f"   title: {ref_patent.title}")
                    print(f"   corresponding MainPatent: {ref_patent.main_patent_id}")
                    if ref_patent.abstract:
                        print(f"   abstract: {ref_patent.abstract[:200]}...")

        if best_idea:
            print("\n" + "="*80)
            print("Best Innovation Idea:")
            print("="*80)
            print(f"\nInnovation Description:\n{best_idea.text}")
            print(f"\nInnovation Claims ({len(best_idea.claims)}个):")
            for i, claim in enumerate(best_idea.claims, 1):
                print(f"  {i}. {claim}")

            print(f"\nEvaluation Metrics:")
            print(
                f"  - Novelty: {self.reward_calculator._novelty_score(best_idea):.3f}")
            print(
                f"  - Feasibility: {self.reward_calculator._feasibility_score(best_idea):.3f}")
            print(f"  - Composite Score: {best_reward:.3f}")

    def close(self):
        self.network.close()
