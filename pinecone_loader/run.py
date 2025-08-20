#!/usr/bin/env python3
"""
Pinecone Loader - Main Script
Processes Word documents and loads them into Pinecone vector database
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from document_reader import read_document
from text_splitter import split_text
from embedding_generator import generate_embeddings
from pinecone_store import store_embeddings


def validate_file_path(file_path):
    """
    Validate that the provided file path exists and is a Word document
    
    Args:
        file_path (str): Path to the Word document
        
    Returns:
        Path: Validated file path
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a Word document
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.suffix.lower() in ['.docx', '.doc']:
        raise ValueError(f"File must be a Word document (.docx or .doc), got: {file_path.suffix}")
    
    return file_path


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        str: File path from arguments
    """
    parser = argparse.ArgumentParser(
        description="Load Word documents into Pinecone vector database"
    )
    
    parser.add_argument(
        'file_path',
        help='Path to the Word document to process'
    )
    
    args = parser.parse_args()
    return args.file_path


def process_document(file_path):
    """
    Main processing pipeline for a Word document
    
    Args:
        file_path (str or Path): Path to the Word document to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"🔍 Starting document processing: {file_path}")
        
        # Step 1: Document Reading
        print("📖 Reading document...")
        document_text = read_document(file_path)
        print(f"✅ Document read successfully. Text length: {len(document_text)} characters")
        
        # Step 2: Text Splitting
        print("✂️  Splitting text into chunks...")
        text_chunks = split_text(document_text)
        print(f"✅ Text split into {len(text_chunks)} chunks")
        
        # Step 3: Generate Embeddings
        print("🔢 Generating embeddings...")
        embeddings = generate_embeddings(text_chunks)
        print(f"✅ Generated embeddings for {len(embeddings)} chunks")
        
        # Step 4: Store in Pinecone
        print("💾 Storing embeddings in Pinecone...")
        result = store_embeddings(embeddings, text_chunks, file_path)
        print(f"✅ Successfully stored {len(embeddings)} embeddings in Pinecone")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        return False


def main():
    """
    Main function to handle document processing pipeline
    """
    try:
        # Parse command line arguments and validate file
        file_path = parse_arguments()
        validated_path = validate_file_path(file_path)
        
        # Process the document
        success = process_document(validated_path)
        
        if success:
            print(f"\n🎉 Successfully processed and loaded: {validated_path.name}")
            sys.exit(0)
        else:
            print(f"\n💥 Failed to process: {validated_path.name}")
            sys.exit(1)
            
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()