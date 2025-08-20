#!/usr/bin/env python3
"""
Embedding Generator Module
Handles generating embeddings from text chunks using OpenAI's embedding model
"""

import sys
from pathlib import Path
import openai

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import EMBEDDING_MODEL, OPENAI_API_KEY, EMBEDDING_DIMENSIONS
except ImportError:
    # Default values if not defined in config
    EMBEDDING_MODEL = "text-embedding-3-small"
    OPENAI_API_KEY = None
    EMBEDDING_DIMENSIONS = 512

# Initialize OpenAI client
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in config.py or environment")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def generate_embeddings(text_chunks):
    """
    Generate embeddings for text chunks using OpenAI's embedding model
    
    Args:
        text_chunks (list): List of text chunks to embed
        
    Returns:
        list: List of embedding vectors
        
    Raises:
        ValueError: If text_chunks is empty or invalid
        Exception: If there's an error generating embeddings
    """
    if not text_chunks:
        raise ValueError("Text chunks list cannot be empty")
    
    if not isinstance(text_chunks, list):
        raise ValueError("text_chunks must be a list")
    
    try:
        print(f"🔢 Generating embeddings for {len(text_chunks)} chunks using {EMBEDDING_MODEL} (dim: {EMBEDDING_DIMENSIONS})")
        
        # Generate embeddings for all chunks in one API call
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text_chunks,
            dimensions=EMBEDDING_DIMENSIONS
        )
        
        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        
        print(f"✅ Successfully generated {len(embeddings)} embeddings")
        return embeddings
        
    except openai.OpenAIError as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")


# For testing purposes
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python embedding_generator.py 'text to embed'")
        print("  python embedding_generator.py 'chunk1' 'chunk2' 'chunk3'")
        sys.exit(1)
    
    try:
        # Get text chunks from command line arguments
        test_chunks = sys.argv[1:]
        
        print(f"🧪 Testing embedding generation")
        print(f"Input chunks: {len(test_chunks)}")
        print(f"Using model: {EMBEDDING_MODEL} (dimensions: {EMBEDDING_DIMENSIONS})")
        print("-" * 50)
        
        # Show input chunks
        for i, chunk in enumerate(test_chunks, 1):
            print(f"Chunk {i} ({len(chunk)} chars): {chunk[:100]}...")
        
        print("-" * 50)
        
        # Generate embeddings
        embeddings = generate_embeddings(test_chunks)
        
        # Show results
        print(f"\n📊 Results:")
        print(f"Generated embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Show sample of first embedding
        print(f"Sample from first embedding: {embeddings[0][:5]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)