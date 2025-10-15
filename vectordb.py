import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class SearchResult:
    score: float
    document: str
    path: str


class VectorDatabase:
    def __init__(self, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 collection_name: str = "search_engine",
                 use_local: bool = False):
        """
        Initialize the VectorDatabase with Qdrant client.
        
        Args:
            url: Qdrant server URL (defaults to env var or localhost for local mode)
            api_key: Qdrant API key (defaults to env var, not required for local)
            collection_name: Name of the collection to use
            use_local: If True, use local Qdrant instance (localhost:6333)
        """
        self.collection_name = collection_name
        self.use_local = use_local
        
        # Configure connection based on mode
        if use_local:
            self.url = "http://localhost:6333"
            self.api_key = None
        else:
            self.url = url or os.getenv("QDRANT_URL")
            self.api_key = api_key or os.getenv("QDRANT_API_KEY")
            
            # Validate cloud configuration
            if not self.url or not self.api_key:
                raise ValueError(
                    "For cloud mode, Qdrant URL and API key are required. "
                    "Set QDRANT_URL and QDRANT_API_KEY environment variables "
                    "or use use_local=True for local testing."
                )
        
        # Initialize Qdrant client with error handling
        try:
            if self.api_key:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self.client = QdrantClient(url=self.url)
            
            mode = "local" if use_local else "cloud"
            print(f"✅ Connected to Qdrant ({mode}) at {self.url}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            if use_local:
                print("💡 Make sure Qdrant is running locally: docker run -p 6333:6333 qdrant/qdrant")
            else:
                print("💡 Check your Qdrant cloud credentials and network connection")
            raise
    
    def create_collection(self, vector_size: int) -> None:
        """Create a new collection with the specified vector size."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection '{self.collection_name}' with vector size {vector_size}")
        except Exception as e:
            print(f"Collection '{self.collection_name}' already exists or error: {e}")
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False
    
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
    
    def clear_collection(self) -> None:
        """Clear all points from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="exists",
                                match=models.MatchValue(value=True)
                            )
                        ]
                    )
                )
            )
            print(f"Cleared all points from collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def upsert_vectors(self, 
                      vectors: List[List[float]], 
                      documents: List[str], 
                      paths: List[str],
                      batch_size: int = 100) -> None:
        """
        Upsert vectors with their associated documents and paths.
        
        Args:
            vectors: List of vector embeddings
            documents: List of document contents
            paths: List of file paths
            batch_size: Number of points to upload per batch
        """
        if not vectors or not documents or not paths:
            print("No data to upsert")
            return
        
        if len(vectors) != len(documents) or len(vectors) != len(paths):
            raise ValueError("Vectors, documents, and paths must have the same length")
        
        # Create points
        points = []
        for i, (vector, document, path) in enumerate(zip(vectors, documents, paths)):
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "document": document,
                    "path": path,
                    "index": i
                }
            ))
        
        # Upload points in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Upserted {len(points)} vectors to collection '{self.collection_name}'")
    
    def search_vectors(self, 
                      query_vector: List[float], 
                      limit: int = 5) -> List[SearchResult]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: The query vector to search with
            limit: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            results = []
            for result in search_results:
                results.append(SearchResult(
                    score=float(result.score),
                    document=result.payload["document"],
                    path=result.payload["path"]
                ))
            
            return results
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []


# Create a default instance for backward compatibility
def get_default_vector_db() -> VectorDatabase:
    """Get a default VectorDatabase instance."""
    return VectorDatabase()


# For testing purposes
if __name__ == "__main__":
    db = VectorDatabase()
    print("Available collections:", db.list_collections())
    print("Collection stats:", db.get_collection_stats())