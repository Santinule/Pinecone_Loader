# Pinecone Document Loader

A simple pipeline to load Word documents into Pinecone vector database for RAG (Retrieval Augmented Generation) applications.

## What it does

Takes a Word document → Splits into chunks → Generates embeddings → Stores in Pinecone

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your API keys
Create/update `config.py`:
```python
# OpenAI Configuration
OPENAI_API_KEY = "your-openai-api-key"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512  # Match your Pinecone index dimension

# Pinecone Configuration  
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-index-name"

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### 3. Process a document
```bash
python pinecone_loader/run.py path/to/your-document.docx
```

## Project Structure
```
.
├── .env
├── config.py                    # API keys and configuration
├── requirements.txt             # Python dependencies
├── structure.txt
└── utilities/
    └── test_connections.py
└── pinecone_loader/
    ├── __init__.py
    ├── run.py                   # Main script - entry point
    ├── document_reader.py       # Document Reading functionality
    ├── text_splitter.py        # Text Splitting functionality
    ├── embedding_generator.py  # Generate Embeddings functionality
    └── pinecone_store.py       # Store into Pinecone functionality
```

## Usage Examples

### Process a single document
```bash
python pinecone_loader/run.py docs/my-document.docx
```

### Check Pinecone index stats
```bash
python pinecone_loader/pinecone_store.py stats
```

### Test individual components
```bash
# Test document reading
python pinecone_loader/document_reader.py docs/my-document.docx

# Test text splitting  
python pinecone_loader/text_splitter.py "Your text content here"

# Test embedding generation
python pinecone_loader/embedding_generator.py "Sample text chunk"

# Test Pinecone storage
python pinecone_loader/pinecone_store.py test docs/my-document.docx
```

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone account with existing index
- Word documents (.docx format)

## Notes

- Automatically handles accented characters in filenames
- Generates ASCII-only vector IDs for Pinecone compatibility
- Includes rich metadata (source file, timestamps, chunk info)
- Optimized for text-embedding-3-small model
- Configurable chunk sizes and overlap for your use case
