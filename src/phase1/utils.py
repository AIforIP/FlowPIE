from pyexpat import model

from neo4j import GraphDatabase
from .models import *
from openai import OpenAI
import json
from typing import List, Tuple
from config.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, TEMPERATURE,
    KEYWORDS_COUNT_MIN, KEYWORDS_COUNT_MAX,
    TOKEN_LOG_ENABLED, TOKEN_LOG_PATH1
)
import time
import os
import json as _json
import tiktoken

def token_count(
            messages,
            model="gpt-4o"
        ):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    tokens = len(encoding.encode(messages))
    return tokens



def evaluator(prompt: str, model: str = OPENAI_MODEL, token_log_path: str = TOKEN_LOG_PATH1) -> str:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    completion = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {"role": "system", "content": "You are a patent text parsing assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    try:
        usage = {}
        if hasattr(completion, 'usage') and completion.usage is not None:
            # some SDKs provide usage as dict-like
            usage = {k: int(v) for k, v in completion.usage.items()}
        else:
            # fallback: try dict access
            usage = dict(getattr(completion, 'usage', {}) or {})
    except Exception:
        usage = {}

    # Append token log as JSONL
    try:
        if token_log_path:
            os.makedirs(os.path.dirname(token_log_path), exist_ok=True)
            record = {
                'ts': int(time.time()),
                'model': model,
                'input_tokens': token_count(prompt),
                'output_tokens': token_count(completion.choices[0].message.content),
                'usage': usage,
                'prompt_snippet': prompt[:200],
                
            }
            with open(token_log_path, 'a', encoding='utf-8') as fh:
                fh.write(_json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass

    return completion.choices[0].message.content

def generator(prompt: str, model: str = OPENAI_MODEL, token_log_path: str = TOKEN_LOG_PATH1) -> str:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    completion = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {"role": "system", "content": "You are a patent text parsing assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    try:
        usage = {}
        if hasattr(completion, 'usage') and completion.usage is not None:
            # some SDKs provide usage as dict-like
            usage = {k: int(v) for k, v in completion.usage.items()}
        else:
            # fallback: try dict access
            usage = dict(getattr(completion, 'usage', {}) or {})
    except Exception:
        usage = {}

    # Append token log as JSONL
    try:
        if TOKEN_LOG_ENABLED:
            os.makedirs(os.path.dirname(token_log_path), exist_ok=True)
            record = {
                'ts': int(time.time()),
                'model': model,
                'input_tokens': token_count(prompt),
                'output_tokens': token_count(completion.choices[0].message.content),
                'usage': usage,
                'prompt_snippet': prompt[:200],
                
            }
            with open(token_log_path, 'a', encoding='utf-8') as fh:
                fh.write(_json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass

    return completion.choices[0].message.content


def extract_keywords_from_query(query: str) -> List[str]:
    """
Extract keywords from user query

Args:
    query: User query text

Returns:
    List of keywords
    """
    prompt = f"""Please extract {KEYWORDS_COUNT_MIN}-{KEYWORDS_COUNT_MAX} of the most critical technical keywords from the following user queries for patent search.

    User Query: "{query}"

    Requirements:
    1. Extract technical terms, core concepts, and functional descriptions
    2. Keywords should be specific and valuable for search
    3. Return format as a JSON array, for example: ["keyword1", "keyword2", ...]
    4. Only return the JSON array, no other explanations

    Keywords:"""
    
    try:
        response = generator(prompt, "gpt-4o-mini")
        
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        keywords = json.loads(response)
        
        if isinstance(keywords, list):
            print(f"Extracted keywords: {keywords}")
            return keywords
        else:
            print("Keyword format error, using simple tokenization")
            return query.split()
            
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return [word.strip() for word in query.split() if len(word.strip()) > 2]


def create_fulltext_index(uri: str = NEO4J_URI, 
                         user: str = NEO4J_USER, 
                         password: str = NEO4J_PASSWORD) -> None:
    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_lifetime=3600,
        connection_acquisition_timeout=120,
        keep_alive=True,
    )
    
    with driver.session() as session:
        try:
            result = session.run("SHOW INDEXES")
            indexes = [record['name'] for record in result]
            
            if 'referencePatentFulltext' not in indexes:
                print("referencePatentFulltext")
                session.run("""
                    CREATE FULLTEXT INDEX referencePatentFulltext IF NOT EXISTS
                    FOR (rp:ReferencePatent)
                    ON EACH [rp.title, rp.abstract, rp.claims]
                """)
            else:
                print("referencePatentFulltext index already exists")
                
        except Exception as e:
            print(f"Error when creating full-text index: {e}")
    
    driver.close()


def save_results_to_json(datalist: list, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def load_test_data(data_path: str) -> list:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)
