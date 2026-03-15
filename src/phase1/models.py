from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import math
import numpy as np
from neo4j import GraphDatabase
import json


@dataclass
class Patent:
    """Main Patent Structure"""
    id: str
    title: str
    abstract: str
    claims: List[str]
    embedding: np.ndarray
    claim_blocks: List[Dict]
    
@dataclass
class ReferencePatent:
    """Reference Patent Structure"""
    id: str
    title: str
    abstract: str
    claims: str
    main_patent_id: str
    embedding: np.ndarray
    
@dataclass
class GeneratedIdea:
    """Generated Idea Structure"""
    text: str
    embedding: np.ndarray
    source_patents: List[str]
    claims: List[str]
    reward: float = 0.0


class MCTSNode:
    """MCTS Tree Node Structure"""
    def __init__(self, ref_patent_ids: List[str], parent=None, action=None):
        self.ref_patent_ids = ref_patent_ids
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        
    def ucb_score(self, c=1.414):
        """UCB Score Calculation"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.avg_reward
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c=1.414):
        """Select the child node with UCB score."""
        return max(self.children, key=lambda child: child.ucb_score(c))
    
    def add_child(self, ref_patent_id: str):
        """Child Node Addition."""
        new_ref_patent_ids = self.ref_patent_ids + [ref_patent_id]
        child = MCTSNode(new_ref_patent_ids, parent=self, action=ref_patent_id)
        self.children.append(child)
        return child
    
    def update(self, reward: float):
        """Node Statistics Update."""
        self.visits += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.visits


class PatentSemanticNetwork:
    """Patent Semantic Network"""
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=120,
            keep_alive=True,
        )
        self.ref_patents_cache = {}
        self.main_patents_cache = {}
        
    def close(self):
        self.driver.close()
    
    def hybrid_search_reference_patents(
        self,
        query_text: str,
        keywords: List[str],
        query_embedding: List[float],
        limit: int = 15,
        alpha: float = 0.6,
        min_semantic_score: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid Retrieval ReferencePatent:
        - Vector semantic similarity retrieval (based on query embedding)
        - Full-text keyword matching retrieval (based on extracted keywords)
        - Weighted fusion ranking of both

        Args:
            query_text: Original query text
            keywords: List of extracted keywords
            query_embedding: Vector representation of the query
            limit: Number of results to return
            alpha: Vector similarity weight (0-1), 1-alpha is the keyword weight
            min_semantic_score: Minimum semantic similarity threshold

        Returns:
            List of retrieved patents (including ID and score)
        """
        with self.driver.session() as session:
            keyword_query = " OR ".join(keywords)
            
   
            cypher = """
            // Step 1: Vector Similarity Retrieval
            CALL db.index.vector.queryNodes(
                'patent_abstract_vector',
                $limit * 3,
                $query_embedding
            ) YIELD node AS vec_node, score AS semantic_score
            WHERE vec_node:ReferencePatent AND semantic_score >= $min_semantic_score
            
            // Step 2: Full-text Keyword Retrieval
            WITH collect({node: vec_node, semantic_score: semantic_score}) AS vector_results
            
            CALL db.index.fulltext.queryNodes(
                'referencePatentFulltext',
                $keyword_query
            ) YIELD node AS text_node, score AS text_score
            WHERE text_node:ReferencePatent
            
            WITH vector_results, collect({node: text_node, text_score: text_score}) AS text_results
            
            // Step 3: Fuse the two retrieval results
            UNWIND vector_results AS vr
            WITH vr, text_results
            
            // Find the score in the text retrieval
            WITH vr.node AS node, 
                 vr.semantic_score AS semantic_score,
                 [tr IN text_results WHERE tr.node = vr.node | tr.text_score][0] AS text_score
            
            // Calculate the fused score
            WITH node,
                 semantic_score,
                 COALESCE(text_score, 0.0) AS text_score,
                 ($alpha * semantic_score + (1 - $alpha) * COALESCE(text_score, 0.0)) AS final_score
            
            RETURN node.id AS id,
                   node.title AS title,
                   node.abstract AS abstract,
                   semantic_score,
                   text_score,
                   final_score
            ORDER BY final_score DESC
            LIMIT $limit
            """
            
            try:
                result = session.run(
                    cypher,
                    query_embedding=query_embedding,
                    keyword_query=keyword_query,
                    limit=limit,
                    alpha=alpha,
                    min_semantic_score=min_semantic_score
                )
                
                patents = []
                for record in result:
                    patents.append({
                        'id': record['id'],
                        'title': record['title'],
                        'abstract': record['abstract'],
                        'semantic_score': record['semantic_score'],
                        'text_score': record['text_score'],
                        'final_score': record['final_score']
                    })
                
                print(f"\nHybrid search results: Found {len(patents)} patents")
                if patents:
                    print("\nTop-5:")
                    for i, p in enumerate(patents[:3], 1):
                        print(f"  {i}. [{p['final_score']:.3f}] {p['title'][:60]}...")
                
                return patents
                
            except Exception as e:
                print(f"Mixed retrieval error: {e}")
                print("Try using an alternative retrieval scheme....")
                return self._fallback_search(session, query_embedding, keywords, limit)
    
    def _fallback_search(self, session, query_embedding: List[float], 
                         keywords: List[str], limit: int) -> List[Dict]:
        """Fallback Retrieval Plan (when vector index is unavailable)"""
        
        cypher = """
        MATCH (rp:ReferencePatent)
        WHERE rp.abstract_embedding IS NOT NULL
        WITH rp,
             gds.similarity.cosine(rp.abstract_embedding, $query_embedding) AS similarity
        WHERE similarity >= 0.3
        RETURN rp.id AS id,
               rp.title AS title,
               rp.abstract AS abstract,
               similarity AS semantic_score,
               0.0 AS text_score,
               similarity AS final_score
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        try:
            result = session.run(cypher, 
                               query_embedding=query_embedding, 
                               limit=limit)
            
            patents = []
            for record in result:
                patents.append({
                    'id': record['id'],
                    'title': record['title'],
                    'abstract': record.get('abstract', ''),
                    'semantic_score': record['semantic_score'],
                    'text_score': record['text_score'],
                    'final_score': record['final_score']
                })
            
            print(f"Fallback plan found {len(patents)} patents")
            return patents
            
        except Exception as e:
            print(f"Fallback plan also failed: {e}")
            return []
    
    def get_related_reference_patents(self, ref_patent_id: str, 
                                     limit=10,
                                     exclude_ids: Set[str] = None) -> List[Tuple[str, float]]:
        """Retrieve other Reference Patents related to the specified Reference Patent"""
        if exclude_ids is None:
            exclude_ids = set()
        
        with self.driver.session() as session:
            query = """
            MATCH (p1:ReferencePatent)-[r]-(p2:ReferencePatent)
            WHERE p1.id = $ref_patent_id AND p2.id <> $ref_patent_id
            RETURN p2.id as ref_id, 
                   CASE 
                     WHEN r.similarity IS NOT NULL THEN r.similarity
                     WHEN r.weight IS NOT NULL THEN r.weight
                     ELSE 0.5
                   END as similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            result = session.run(query, ref_patent_id=ref_patent_id, limit=limit)
            
            related = [(record['ref_id'], record['similarity']) 
                      for record in result 
                      if record['ref_id'] not in exclude_ids]
            
            return related
    
    def get_reference_patent(self, ref_patent_id: str) -> ReferencePatent:
        """Retrieve detailed information of a ReferencePatent (with caching)"""
        if ref_patent_id in self.ref_patents_cache:
            return self.ref_patents_cache[ref_patent_id]
        
        with self.driver.session() as session:
            query = """
            MATCH (rp:ReferencePatent)
            WHERE rp.id = $ref_patent_id
            RETURN rp.id as id, rp.title as title, rp.abstract as abstract,
                   rp.claims as claims, rp.main_patent_id as main_patent_id,
                   rp.abstract_embedding as embedding
            """
            result = session.run(query, ref_patent_id=ref_patent_id)
            record = result.single()
            
            if record:
                embedding_data = record.get('embedding')
                if embedding_data:
                    if isinstance(embedding_data, str):
                        embedding = np.array(json.loads(embedding_data))
                    else:
                        embedding = np.array(embedding_data)
                else:
                    embedding = np.random.randn(1024) 
                
                ref_patent = ReferencePatent(
                    id=record['id'],
                    title=record.get('title', ''),
                    abstract=record.get('abstract', ''),
                    claims=record.get('claims', ''),
                    main_patent_id=record.get('main_patent_id', ''),
                    embedding=embedding
                )
                self.ref_patents_cache[ref_patent_id] = ref_patent
                return ref_patent
        
        return None
    
    def get_main_patent(self, main_patent_id: str) -> Patent:
        """Retrieve detailed information of a MainPatent (with caching)"""
        if main_patent_id in self.main_patents_cache:
            return self.main_patents_cache[main_patent_id]
        
        with self.driver.session() as session:
            query = """
            MATCH (mp:MainPatent)
            WHERE mp.id = $main_patent_id
            RETURN mp.id as id, mp.title as title, mp.abstract as abstract,
                   mp.claims as claims, mp.abstract_embedding as embedding,
                   mp.blocks as claim_blocks
            """
            result = session.run(query, main_patent_id=main_patent_id)
            record = result.single()
            
            if record:
                embedding_data = record.get('embedding')
                if embedding_data:
                    if isinstance(embedding_data, str):
                        try:
                            embedding = np.array(json.loads(embedding_data))
                        except:
                            embedding = np.random.randn(1024)
                    elif isinstance(embedding_data, list):
                        embedding = np.array(embedding_data)
                    else:
                        embedding = np.random.randn(1024)
                else:
                    embedding = np.random.randn(1024)
                
   
                claims_data = record.get('claims')
                claims = []
                
                if claims_data is None or claims_data == '':
                    claims = []
                elif isinstance(claims_data, list):
                    claims = claims_data
                elif isinstance(claims_data, str):
                    claims_data = claims_data.strip()
                    if claims_data and claims_data not in ['', '[]', 'null']:
                        try:
                            claims = json.loads(claims_data)
                            if not isinstance(claims, list):
                                claims = [str(claims)]
                        except json.JSONDecodeError:
                            claims = [claims_data]
                    else:
                        claims = []
                else:
                    claims = [str(claims_data)]
                
                blocks_data = record.get('claim_blocks')
                claim_blocks = []
                
                if blocks_data is None or blocks_data == '':
                    claim_blocks = []
                elif isinstance(blocks_data, list):
                    claim_blocks = blocks_data
                elif isinstance(blocks_data, str):
                    blocks_data = blocks_data.strip()
                    if blocks_data and blocks_data not in ['', '[]', 'null']:
                        try:
                            claim_blocks = json.loads(blocks_data)
                            if not isinstance(claim_blocks, list):
                                claim_blocks = []
                        except json.JSONDecodeError:
                            claim_blocks = []
                    else:
                        claim_blocks = []
                else:
                    claim_blocks = []
                
                main_patent = Patent(
                    id=record['id'],
                    title=record.get('title', ''),
                    abstract=record.get('abstract', ''),
                    claims=claims,
                    embedding=embedding,
                    claim_blocks=claim_blocks
                )
                self.main_patents_cache[main_patent_id] = main_patent
                return main_patent
        
        return None
