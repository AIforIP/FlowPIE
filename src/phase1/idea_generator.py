import random
import ast
import numpy as np
from typing import List, Optional
from neo4j import GraphDatabase
from .models import ReferencePatent, GeneratedIdea
from .utils import evaluator, generator
from config.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CLAIM_CROSSOVER_SAMPLE_SIZE, MUTATION_RATE,
    EXTRA_FEATURES_SAMPLE_SIZE
)


class IdeaGenerator:
    """Idea genetor"""
    
    @staticmethod
    def semantic_crossover(ref_patents: List[ReferencePatent], 
                          weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Semantic Cross - Generate a fused embedding by weighted combination of multiple patent embeddings
            
            Args:
                ref_patents: List of reference patents
                weights: List of weights, if None, uniform weights are used
                
            Returns:
                The fused normalized embedding vector
        """
        if weights is None:
            weights = [1.0 / len(ref_patents)] * len(ref_patents)
        
        embeddings = [rp.embedding for rp in ref_patents]
        weighted_embedding = np.zeros_like(embeddings[0])
        
        for w, emb in zip(weights, embeddings):
            weighted_embedding += w * emb
        

        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding /= norm
            
        return weighted_embedding
    
    @staticmethod
    def claim_crossover(ref_patents: List[ReferencePatent]) -> List[str]:
        """
        Right Requirement Intersection - Extract and combine features from multiple patent knowledge graphs
        
        Args:
            ref_patents: List of reference patents
            
        Returns:
            List of combined features
        """
        driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600,
                connection_acquisition_timeout=120,
                keep_alive=True,
        )
        raw_elements = []

        with driver.session() as session:
            for rp in ref_patents:
                query = """
                MATCH (rp:ReferencePatent {id: $rid})
                OPTIONAL MATCH (rp)-[:HAS_FUNCTION]->(f:Function)
                OPTIONAL MATCH (rp)-[:HAS_COMPONENT]->(c:Component)
                OPTIONAL MATCH (rp)-[:HAS_INNOVATION]->(i:Innovation)
                RETURN 
                    collect(DISTINCT f.name) AS functions,
                    collect(DISTINCT c.name) AS components,
                    collect(DISTINCT i.name) AS innovations
                """

                result = session.run(query, rid=rp.id).single()

                functions = [x for x in result["functions"] if x]
                components = [x for x in result["components"] if x]
                innovations = [x for x in result["innovations"] if x]

                selected = []
                if functions:
                    selected.extend(random.sample(functions, min(1, len(functions))))
                if components:
                    selected.extend(random.sample(components, min(1, len(components))))
                if innovations:
                    selected.extend(random.sample(innovations, min(1, len(innovations))))

                raw_elements.extend(selected)

        driver.close()
        
        raw_elements = list(dict.fromkeys(raw_elements))[:CLAIM_CROSSOVER_SAMPLE_SIZE]


        prompt = f"""
        You are an expert in cross-domain innovation and semantic feature recombination. 
        You will receive a list of structural or functional elements extracted from multiple reference patents. 
        Your goal is to perform "evolution-style crossover," similar to the idea of:
        "Natural Language Processing + Deep Learning -> ChatGPT."

        In other words, you must not simply combine original elements —  
        you must introduce **new features** that were *not present* in the input, enabling the system to enter a **new capability space**.

        ### Input Elements
        {raw_elements}

        ### Your Task
        Generate **5–8 innovative feature fragments**, each representing a new combination derived from the given elements, with the following requirements:

        1. Each fragment must integrate **multiple input elements** in a meaningful way.  
        2. Each fragment must introduce **at least one new feature** that does *not* appear in the original list  
        (e.g., new sensing ability, adaptive behavior, reconfigurable structure, dual-mode operation, energy recovery, biological analogies, etc.).  
        3. The newly introduced feature should expand the system's capability domain, similar to how  
        "adding a new organ or function" enables evolutionary transitions.  
        4. The output must be a **Python list of strings**, with no explanations before or after.

        ### Examples (just illustrative):
        - "Combining the structural stabilizer with the adaptive actuator to introduce a dual-environment self-modulation mechanism"
        - "Integrating the thermal module with the optical feedback unit to introduce cross-modal environmental sensing"
        - "Merging attachment components with micro-texture modules to enable bio-inspired self-repairing functionality"

        ### Now produce 5–8 evolved crossover feature fragments in Python list format:

        """
        try:
            model_output = generator(prompt).strip()
            new_elements = ast.literal_eval(model_output)
            if isinstance(new_elements, list):
                return new_elements[:8]
            else:
                return raw_elements[:8]
        except Exception:
            return raw_elements[:8]

    @staticmethod
    def semantic_mutation(claims: List[str], ref_patents: List[ReferencePatent], mutation_rate: float = MUTATION_RATE) -> List[str]:
        """
        Semantic Variation - Evolve the claims to generate new features

        Args:
            claims: List of original claims
            mutation_rate: Mutation rate (0-1)

        Returns:
            List of mutated claims
        """
        mutated_claims = []
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            connection_acquisition_timeout=120,
            keep_alive=True,
        )
        extra_features = []

        ref_patent_ids = [rp.id for rp in ref_patents]

        with driver.session() as session:
            query = """
            MATCH (n)
            WHERE n:Function OR n:Component OR n:Innovation
            OPTIONAL MATCH (n)-[:HAS_REFERENCE_PATENT]->(rp)
            WITH n, rp
            WHERE rp IS NULL OR NOT rp.id IN $ref_patent_ids
            RETURN collect(DISTINCT n.name) AS names
            """
            result = session.run(query, ref_patent_ids=ref_patent_ids).single()
            extra_features = [x for x in result["names"] if x]
            random.shuffle(extra_features)
            
            extra_features = extra_features[:40]

        driver.close()

        for claim in claims:
            if random.random() >= mutation_rate:
                mutated_claims.append(claim)
                continue

            sampled_features = random.sample(extra_features, min(EXTRA_FEATURES_SAMPLE_SIZE, len(extra_features)))

            prompt = f"""
            You are a cross-domain innovation mutation engine. 
            Your goal is to take an existing claim fragment and produce an evolved version 
            that introduces **new features** or **fixes its limitations**, similar to biological mutation.

            ### Original Claim Fragment
            {claim}

            ### Additional Feature Pool (from patent knowledge graph)
            {sampled_features}

            ### Mutation Requirements
            1. Produce a **single mutated version** of the claim.
            2. The mutation must:
            - Introduce at least **one new feature** not present in the original claim.
            - Or modify an existing feature to solve a limitation or deficiency.
            - Optionally incorporate items from the feature pool.
            3. The result must be **concise** but technically meaningful.
            4. Output ONLY the mutated text string, with no explanation.

            ### Now output the mutated claim fragment:
            """

            try:
                mutated = generator(prompt).strip()
                mutated_claims.append(mutated)
            except Exception:
                mutated_claims.append(claim)

        return mutated_claims
    
    @classmethod
    def generate_idea(cls, ref_patents: List[ReferencePatent], query: str) -> GeneratedIdea:
        """
        Generate innovative ideas - Integrate cross and mutation operations to generate complete creativity

        Args:
            ref_patents: List of reference patents
            query: User query

        Returns:
            Generated innovative idea object
        """
        fused_embedding = cls.semantic_crossover(ref_patents)
        combined_claims = cls.claim_crossover(ref_patents)
        mutated_claims = cls.semantic_mutation(combined_claims, ref_patents=ref_patents, mutation_rate=MUTATION_RATE)
        idea_text = cls._generate_text_from_claims(mutated_claims, combined_claims, ref_patents, query)
        
        return GeneratedIdea(
            text=idea_text,
            embedding=fused_embedding,
            source_patents=[rp.id for rp in ref_patents],
            claims=mutated_claims
        )
    
    @staticmethod
    def _generate_text_from_claims(claims: List[str], combined_claims: List[str], 
                                   ref_patents: List[ReferencePatent], query: str) -> str:
        """
        Generate technical solution text from claims

        Args:
            claims: Modified claims
            combined_claims: Crossed claims
            ref_patents: List of reference patents
            query: User query

        Returns:
            Generated technical solution text
        """
        ref_info = "\n".join([
            f"- {rp.title}: {rp.abstract if rp.abstract else 'no abstract'}"
            for rp in ref_patents
        ]) or "no reference"

        claims_text = "\n".join([f"- {c}" for c in claims]) or "(no claims)"

        prompt = f"""
        You are an expert in cross-domain innovation and idea generation. Your task is to generate a scientific idea that aligns with the user’s scientific intent. The objective is to produce ideas that are technically specific, experimentally verifiable, and potentially claim-supportive rather than abstract or purely descriptive. The proposed idea should address a clear scientific problem, produce a measurable and verifiable technical effect, and demonstrate non-trivial scientific value that could be considered meaningful in a paper peer review context.

        Your goal is NOT to be abstractly impressive, but to be **technically specific, testable,
        and claim-supportive**.

        Vague, generic, or purely descriptive ideas will be considered FAILURE.

        Your primary objective is to propose an idea that:
        - Solves a clear technical problem the user truly cares about,
        - Produces a **measurable and verifiable technical effect**,
        - Would be considered **non-trivial and valuable** by a patent examiner.
        
        ### The User Query
        {query}

        The reference provided below are **NOT the goal**.
        They serve only as **inspirational background and technical constraints**.
        Do NOT summarize, paraphrase, or replicate their claims or structures.


        ### Core Output Requirements

        You must generate:
        1. **A single high-level patent-oriented conceptual idea**
        - Focus on *what the invention fundamentally achieves*,
        - Avoid low-level implementation details,
        - Emphasize why this idea is *effective* rather than merely novel.

        2. **A Theoretical Experimental Validation Plan**
        - Be **rigorous, concrete, and technically grounded**,
        - Clearly define:
            - What technical effect is being validated,
            - What variables are controlled or compared,
            - What measurable indicators demonstrate effectiveness,
        - The validation logic should be strong enough to support patent claims,
            even if the experiment is hypothetical.


        ### Reference Information (for context only)
        Use the following information **only as auxiliary knowledge** to:
        - Inspire feasible directions,
        - Avoid already-saturated solution patterns,
        - Ensure technical plausibility.

        **Do NOT treat them as design templates.**

        - Extracted technical elements (functions, components, innovations):
        {combined_claims}

        - Reference patent metadata:
        {ref_info}

        - Reference claims (high-level context only; do NOT reuse language or structure):
        {claims_text}

        ### REQUIREMENTS
        1. **Conceptual Nature:** Do NOT describe step-by-step engineering instructions or manufacturing recipes. Focus on concrete functional principles.
        2. **Experimental Logic:** You MUST provide a structured experimental design that tests the proposed concept against a baseline (prior art).
        3.You MUST output **ALL** of the following sections (A–E).
        Each section must be **technically concrete**, not rhetorical,include:
            - (A) **Core Concept Description**
            - (B) **Conceptual Functional Principle** 
            - (C) **High-Level Conceptual Workflow**
            - (D) **Potential Innovation Directions**
            - (E) **Theoretical Experimental Design** 

        ### CONTENT GUIDANCE

        (A) **Core Concept Description**
        Briefly summarize the proposed idea and the scientific problem it addresses.

        (B) **Conceptual Functional Principle **
        Describe the key scientific and functional principle. When appropriate, include 1–2 formulas or models representing the underlying logic.

        (C) **High-Level Conceptual Workflow**
        Provide a concrete workflow describing the operation of the proposed idea.

        (D) **Potential Innovation Directions**
        List 3–5 potential scientific direction for future work.

        (E) **Theoretical Experimental Design** (CRITICAL SECTION)
        Design a experiment plan to validate the effect of the idea compared with baselines. The design could specify: 
        Experimental Setup (such as Backbone Models, hyperparameter and so on), 
        Variables (Independent variables, Dependent variables), 
        Evaluation Metrics and Baselines.

        ### OUTPUT
        Provide ONLY the content in the above five sections (A–E).
        Ensure the Experimental Design (Section E) is logical and clearly links the Innovation (Section A/B) to tangible technical benefits.
        """

        try:
            result = generator(prompt).strip()
            if len(result) < 50:
                raise ValueError("text is too short")
            return result
        except Exception:
            text = f"Based on {len(ref_patents)} patents, a novel combination."
            if claims:
                text += f" Core innovation: {', '.join(claims[:3])}."
            return text
