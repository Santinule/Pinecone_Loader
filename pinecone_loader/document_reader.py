#!/usr/bin/env python3
"""
Document Reader Module
Handles reading and extracting text content from Word documents
"""

import sys
from pathlib import Path
from docx import Document


def read_document(file_path):
    """
    Read a Word document and extract its text content
    
    Args:
        file_path (str or Path): Path to the Word document
        
    Returns:
        str: Extracted text content from the document
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error reading the document
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Open and read the Word document
        document = Document(file_path)
        
        # Extract text from all paragraphs
        text_content = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:  # Only add non-empty paragraphs
                text_content.append(text)
        
        # Join paragraphs with newlines
        full_text = '\n'.join(text_content)
        
        if not full_text.strip():
            raise ValueError(f"No text content found in document: {file_path.name}")
        
        return full_text
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise Exception(f"Error reading document {file_path.name}: {str(e)}")


# For testing purposes
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python document_reader.py <path_to_document.docx>")
        sys.exit(1)
    
    try:
        file_path = sys.argv[1]
        text = read_document(file_path)
        print(f"Document: {Path(file_path).name}")
        print(f"Text length: {len(text)} characters")
        print("-" * 50)
        print(text[:500] + "..." if len(text) > 500 else text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)