#!/usr/bin/env python3
"""
Simple test script to verify Ollama connection and functionality.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
import os

def test_ollama_connection():
    """Test Ollama connection and basic functionality."""
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