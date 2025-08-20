#!/usr/bin/env python3
"""
Pinecone Store Module
Handles storing embeddings in existing Pinecone index
"""

import sys
import uuid
import unicodedata
from pathlib import Path
from datetime import datetime
from pinecone import Pinecone

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
except ImportError:
    # Default values if not defined in config
    PINECONE_API_KEY = None
    PINECONE_INDEX_NAME = None

# Initialize Pinecone
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY must be set in config.py")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME must be set in config.py")

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing index
index = pc.Index(PINECONE_INDEX_NAME)


def create_metadata(text_chunk, file_path, chunk_index):
    """
    Create metadata for a text chunk
    
    Args:
        text_chunk (str): The text content
        file_path (str or Path): Source file path
        chunk_index (int): Index of this chunk in the document
        
    Returns:
        dict: Metadata dictionary
    """
    file_path = Path(file_path)
    
    metadata = {
        "text": text_chunk,
        "source": file_path.name,
        "file_path": str(file_path),
        "chunk_index": chunk_index,
        "chunk_length": len(text_chunk),
        "timestamp": datetime.now().isoformat(),
        "document_type": "word_document"
    }
    
    return metadata


def generate_vector_id(file_path, chunk_index):
    """
    Generate a unique ID for a vector (ASCII-only for Pinecone compatibility)
    
    Args:
        file_path (str or Path): Source file path
        chunk_index (int): Index of this chunk
        
    Returns:
        str: Unique vector ID (ASCII-only)
    """
    file_path = Path(file_path)
    
    # Get filename without extension
    base_name = file_path.stem
    
    # Convert to ASCII-only by removing accents and special characters
    # First normalize to decompose accented characters
    normalized = unicodedata.normalize('NFD', base_name)
    
    # Keep only ASCII characters (removes accents)
    ascii_name = ''.join(c for c in normalized if ord(c) < 128)
    
    # Replace any remaining non-alphanumeric characters with underscores
    clean_name = ''.join(c if c.isalnum() else '_' for c in ascii_name)
    
    # Remove multiple consecutive underscores
    clean_name = '_'.join(filter(None, clean_name.split('_')))
    
    # Create ID
    vector_id = f"{clean_name}_chunk_{chunk_index}"
    
    return vector_id


def store_embeddings(embeddings, text_chunks, file_path):
    """
    Store embeddings in existing Pinecone index
    
    Args:
        embeddings (list): List of embedding vectors
        text_chunks (list): List of text chunks corresponding to embeddings
        file_path (str or Path): Source file path for metadata
        
    Returns:
        dict: Storage results with success/failure info
        
    Raises:
        ValueError: If embeddings and text_chunks don't match
        Exception: If there's an error storing in Pinecone
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")
    
    if not text_chunks:
        raise ValueError("Text chunks list cannot be empty")
    
    if len(embeddings) != len(text_chunks):
        raise ValueError(f"Embeddings count ({len(embeddings)}) must match text chunks count ({len(text_chunks)})")
    
    try:
        print(f"💾 Storing {len(embeddings)} embeddings in Pinecone index '{PINECONE_INDEX_NAME}'")
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        
        for i, (embedding, text_chunk) in enumerate(zip(embeddings, text_chunks)):
            # Generate unique ID for this vector
            vector_id = generate_vector_id(file_path, i)
            
            # Create metadata
            metadata = create_metadata(text_chunk, file_path, i)
            
            # Create vector tuple (id, values, metadata)
            vector = (vector_id, embedding, metadata)
            vectors_to_upsert.append(vector)
        
        # Upsert vectors to Pinecone
        upsert_response = index.upsert(vectors=vectors_to_upsert)
        
        # Check upsert response
        upserted_count = upsert_response.get('upserted_count', 0)
        
        print(f"✅ Successfully stored {upserted_count} vectors in Pinecone")
        
        # Return results
        result = {
            "success": True,
            "vectors_stored": upserted_count,
            "index_name": PINECONE_INDEX_NAME,
            "file_source": Path(file_path).name,
            "vector_ids": [generate_vector_id(file_path, i) for i in range(len(embeddings))]
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Error storing embeddings in Pinecone: {str(e)}")


def get_index_stats():
    """
    Get statistics about the Pinecone index
    
    Returns:
        dict: Index statistics
    """
    try:
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        raise Exception(f"Error getting index stats: {str(e)}")


def query_similar_vectors(query_vector, top_k=5, include_metadata=True):
    """
    Query for similar vectors in the index
    
    Args:
        query_vector (list): Query embedding vector
        top_k (int): Number of similar vectors to return
        include_metadata (bool): Whether to include metadata in results
        
    Returns:
        dict: Query results
    """
    try:
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=include_metadata
        )
        return response
    except Exception as e:
        raise Exception(f"Error querying Pinecone: {str(e)}")


# For testing purposes
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pinecone_store.py stats                    # Show index statistics")
        print("  python pinecone_store.py test 'file.docx'         # Test with sample data")
        sys.exit(1)
    
    try:
        command = sys.argv[1]
        
        if command == "stats":
            print("📊 Getting Pinecone index statistics...")
            stats = get_index_stats()
            print(f"Index: {PINECONE_INDEX_NAME}")
            print(f"Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"Dimension: {stats.get('dimension', 'Unknown')}")
            
        elif command == "test":
            if len(sys.argv) < 3:
                print("Error: Please provide a filename for testing")
                sys.exit(1)
            
            test_file = sys.argv[2]
            print(f"🧪 Testing Pinecone storage with sample data")
            print(f"Test file: {test_file}")
            
            # Create sample embeddings and text chunks
            sample_embeddings = [
                [0.1] * 512,  # Sample embedding vector (512 dimensions)
                [0.2] * 512   # Another sample embedding vector (512 dimensions)
            ]
            sample_chunks = [
                "This is a test chunk for Pinecone storage",
                "This is another test chunk for verification"
            ]
            
            # Store the sample data
            result = store_embeddings(sample_embeddings, sample_chunks, test_file)
            
            print(f"\n📊 Storage Results:")
            print(f"Success: {result['success']}")
            print(f"Vectors stored: {result['vectors_stored']}")
            print(f"Index: {result['index_name']}")
            print(f"Vector IDs: {result['vector_ids']}")
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)