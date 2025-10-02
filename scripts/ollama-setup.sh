#!/bin/bash

# Ollama Setup Script for AI Therapist Debugging

set -e

echo "🚀 Setting up Ollama for AI Therapist debugging..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service to be ready..."
timeout 60 bash -c 'until curl -s http://localhost:11434/api/tags > /dev/null; do sleep 2; done'

if [ $? -eq 0 ]; then
    echo "✅ Ollama service is ready"
else
    echo "❌ Ollama service failed to start within 60 seconds"
    exit 1
fi

# Check available models
echo "🔍 Checking available models..."
MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "")

# Pull required models if not present
if [[ "$MODELS" != *"llama3.2"* ]]; then
    echo "📥 Pulling llama3.2 model..."
    curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.2:latest"}'
    echo "✅ llama3.2 model pulled"
else
    echo "✅ llama3.2 model already available"
fi

if [[ "$MODELS" != *"nomic-embed-text"* ]]; then
    echo "📥 Pulling nomic-embed-text model..."
    curl -X POST http://localhost:11434/api/pull -d '{"name": "nomic-embed-text:latest"}'
    echo "✅ nomic-embed-text model pulled"
else
    echo "✅ nomic-embed-text model already available"
fi

# Verify models are working
echo "🧪 Testing model availability..."

# Test llama3.2
LLAMA_TEST=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.2:latest",
  "prompt": "Hello",
  "stream": false
}' | jq -r '.response' 2>/dev/null || echo "")

if [[ -n "$LLAMA_TEST" ]]; then
    echo "✅ llama3.2 model working"
else
    echo "❌ llama3.2 model not responding"
fi

# Test nomic-embed-text
EMBED_TEST=$(curl -s http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text:latest",
  "prompt": "Hello"
}' | jq -r '.embedding' 2>/dev/null || echo "")

if [[ -n "$EMBED_TEST" ]]; then
    echo "✅ nomic-embed-text model working"
else
    echo "❌ nomic-embed-text model not responding"
fi

echo "🎉 Ollama setup complete!"