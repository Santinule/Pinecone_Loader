import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import openai
from pinecone import Pinecone

# Import config after path is set
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connections():
    """Test API connections"""
    try:
        # Test OpenAI connection
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'connection test'"}],
            max_tokens=10
        )
        logger.info("✓ OpenAI connection successful")
        
        # Test Pinecone connection
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()
        logger.info("✓ Pinecone connection successful")
        
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

if __name__ == "__main__":
    if test_connections():
        print("All connections successful! Ready to proceed.")
    else:
        print("Connection test failed. Please check your API keys and configuration.")