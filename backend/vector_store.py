import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional

class VectorStore:
    def __init__(self, dimension: int = 1536, index_type: str = "L2"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings (1536 for text-embedding-3-small)
            index_type: Type of FAISS index ("L2" for L2 distance, "IP" for inner product)
        """
        self.dimension = dimension
        self.texts = []
        self.metadata = []  # Store additional metadata for each text chunk
        
        # Choose index type
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("index_type must be 'L2' or 'IP'")
        
        print(f"✅ VectorStore initialized with {index_type} index, dimension: {dimension}")

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[dict]] = None):
        """
        Add texts and their embeddings to the store.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata for each text chunk
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("Number of metadata entries must match number of texts")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Validate dimensions
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings_array.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store texts and metadata
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))
        
        print(f"✅ Added {len(texts)} documents. Total documents: {len(self.texts)}")

    def search(self, query_embedding: List[float], k: int = 5, score_threshold: Optional[float] = None) -> List[Tuple[str, float, dict]]:
        """
        Search for similar texts.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Optional threshold for similarity scores
            
        Returns:
            List of tuples (text, score, metadata)
        """
        if len(self.texts) == 0:
            print("⚠️ Vector store is empty")
            return []
        
        # Ensure k doesn't exceed available documents
        k = min(k, len(self.texts))
        
        # Convert query to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Validate dimensions
        if query_array.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_array.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.texts):
                continue
                
            # Apply score threshold if specified
            if score_threshold is not None and distance > score_threshold:
                continue
            
            text = self.texts[idx]
            metadata = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append((text, float(distance), metadata))
        
        print(f"🔍 Found {len(results)} similar documents")
        return results

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.texts),
            "dimension": self.dimension,
            "index_type": type(self.index).__name__
        }

    def save(self, filepath: str):
        """Save the vector store to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save texts and metadata
            with open(f"{filepath}.data", 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            
            print(f"✅ Vector store saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")
            raise

    def load(self, filepath: str):
        """Load the vector store from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load texts and metadata
            with open(f"{filepath}.data", 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
            
            print(f"✅ Vector store loaded from {filepath}")
            print(f"📊 Loaded {len(self.texts)} documents")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            raise

    def clear(self):
        """Clear all data from the vector store."""
        self.index.reset()
        self.texts.clear()
        self.metadata.clear()
        print("🗑️ Vector store cleared")

    def remove_by_indices(self, indices: List[int]):
        """Remove documents by their indices (Note: FAISS doesn't support removal, so this recreates the index)."""
        if not indices:
            return
        
        # Get remaining texts and metadata
        remaining_texts = [text for i, text in enumerate(self.texts) if i not in indices]
        remaining_metadata = [meta for i, meta in enumerate(self.metadata) if i not in indices]
        
        if len(remaining_texts) == 0:
            self.clear()
            return
        
        # We need to rebuild the index since FAISS doesn't support removal
        print("⚠️ Rebuilding index to remove documents...")
        
        # This would require re-embedding all remaining texts
        # For now, just clear and let the user re-add
        self.clear()
        print("❌ Cannot remove specific documents without re-embedding. Vector store cleared.")
        print("💡 To remove specific documents, you'll need to re-add all remaining documents with their embeddings.")

# Example usage and utility functions
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:  # Only break if we're not too far back
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks