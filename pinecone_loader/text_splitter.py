#!/usr/bin/env python3
"""
Text Splitter Module
Handles splitting text into chunks with overlap for context preservation
"""

import sys
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    # Default values if not defined in config
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200


def split_text(document_text):
    """
    Split document text into chunks with overlap for context preservation
    
    Args:
        document_text (str): The text to be split
        
    Returns:
        list: List of text chunks
        
    Raises:
        ValueError: If document_text is empty
        Exception: If there's an error splitting the text
    """
    if not document_text or not document_text.strip():
        raise ValueError("Document text cannot be empty")
    
    try:
        # Initialize RecursiveCharacterTextSplitter with appropriate parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        
        # Split the document into chunks with overlap for context preservation
        chunks = text_splitter.split_text(document_text)
        
        # Filter out empty chunks
        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if not cleaned_chunks:
            raise ValueError("No valid chunks created from document text")
        
        return cleaned_chunks
        
    except Exception as e:
        raise Exception(f"Error splitting text: {str(e)}")


def debug_text_input(text):
    """
    Debug helper to analyze the input text
    
    Args:
        text (str): Input text to analyze
    """
    print("=== DEBUG: Text Input Analysis ===")
    print(f"Text type: {type(text)}")
    print(f"Text length: {len(text)} characters")
    print(f"First 100 characters: '{text[:100]}'")
    print(f"Last 100 characters: '{text[-100:]}'")
    print(f"Contains newlines: {'\\n' in text}")
    print(f"Contains double newlines: {'\\n\\n' in text}")
    print(f"Number of lines: {len(text.split(chr(10)))}")
    print("=" * 40)


def debug_chunks(chunks):
    """
    Debug helper to analyze the generated chunks
    
    Args:
        chunks (list): List of chunks to analyze
    """
    print("=== DEBUG: Chunks Analysis ===")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk)} characters")
        print(f"  First 50 chars: '{chunk[:50]}'")
        print(f"  Last 50 chars: '{chunk[-50:]}'")
        if len(chunk) > 100:
            print(f"  Middle sample: '{chunk[len(chunk)//2-25:len(chunk)//2+25]}'")


def test_with_document_reader(file_path):
    """
    Test text splitter using document reader to get actual content
    
    Args:
        file_path (str): Path to the document file
    """
    try:
        # Import document reader
        from document_reader import read_document
        
        print(f"=== Testing with document: {file_path} ===")
        
        # Read the actual document content
        document_text = read_document(file_path)
        print(f"✅ Document read successfully")
        
        # Debug the document content
        debug_text_input(document_text)
        
        # Split the text
        chunks = split_text(document_text)
        print(f"✅ Text split successfully")
        
        # Debug the chunks
        debug_chunks(chunks)
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        return None


# For testing purposes
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python text_splitter.py 'text content to split'  # Test with raw text")
        print("  python text_splitter.py path/to/document.docx    # Test with document file")
        sys.exit(1)
    
    try:
        input_arg = sys.argv[1]
        
        # Check if it's a file path (ends with .docx or .doc)
        if input_arg.endswith(('.docx', '.doc')):
            print("🔍 Detected document file path - using document reader")
            chunks = test_with_document_reader(input_arg)
        else:
            print("🔍 Treating input as raw text")
            debug_text_input(input_arg)
            chunks = split_text(input_arg)
            debug_chunks(chunks)
        
        if chunks:
            print(f"\n📊 Final Results:")
            print(f"Original input: {input_arg}")
            print(f"Number of chunks: {len(chunks)}")
            print("-" * 50)
            
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i} ({len(chunk)} chars):")
                print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                print("-" * 30)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)