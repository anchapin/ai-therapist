#!/usr/bin/env python3
"""
Simple test script to verify Ollama connection and functionality.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
import os

def test_ollama_connection():
    """Verifies the connection to Ollama and checks core functionalities.

    This function performs two main tests:
    1.  **Embeddings**: It tries to create text embeddings using the
        `nomic-embed-text:latest` model. A success is logged if the
        embedding is generated without errors.
    2.  **Chat Model**: It attempts to invoke the `llama3.2:latest` chat
        model with a simple prompt. A success is logged if a response
        is received without errors.

    The function prints the status of each test to the console.

    Returns:
        bool: True if both the embedding and chat model tests pass,
              False otherwise.
    """
    print("Testing Ollama connection...")

    # Test embeddings
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        test_text = "Hello, this is a test for AI therapist."
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Embeddings working: {len(embedding)} dimensions")
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")
        return False

    # Test chat model
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.7)
        response = llm.invoke("Hello, I'm testing the AI therapist system. Can you help me?")
        print(f"✓ Chat model working: {response.content[:100]}...")
    except Exception as e:
        print(f"✗ Chat model failed: {e}")
        return False

    print("All Ollama tests passed! ✓")
    return True

if __name__ == "__main__":
    test_ollama_connection()