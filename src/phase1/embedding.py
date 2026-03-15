from sentence_transformers import SentenceTransformer
from typing import List
from config.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


class HFEmbeddingEncoder:
    """
    HuggingFace Embedding Encoder

    Using the pre-trained BAAI BGE-M3 model for multilingual text encoding
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
        """
        Initialize the embedding encoder

        Args:
            model_name: Model name or path
            device: Execution device ('cuda' or 'cpu')
        """
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, text: str) -> List[float]:
        """
        Encode a single text

        Args:
            text: Input text

        Returns:
            Normalized embedding vector
        """
        text = text.replace("\n", " ")
        emb = self.model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    
    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Batch encode texts

        Args:
            texts: List of input texts

        Returns:
            List of normalized embedding vectors
        """
        cleaned = [t.replace("\n", " ") for t in texts]
        emb = self.model.encode(cleaned, normalize_embeddings=True)
        return [e.tolist() for e in emb]
