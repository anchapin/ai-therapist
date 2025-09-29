# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv ai-therapist-env
source ai-therapist-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Application Commands
```bash
# Run the main AI therapist application
streamlit run app.py

# Test Ollama connection
python test_ollama.py

# Manually download knowledge files
python download_knowledge.py

# Build vector store from PDFs (legacy - app.py does this automatically)
python build_vectorstore.py
```

### Testing
```bash
# Test Ollama integration
python test_ollama.py

# Manual testing: Run app and verify conversation flow with therapy PDFs
```

## Architecture Overview

This is a **Streamlit-based AI therapist application** that uses **Ollama** for local LLM inference and **LangChain** for RAG (Retrieval-Augmented Generation) capabilities.

### Core Components

**Knowledge Downloader (`download_knowledge.py`)**
- Downloads therapy resources from URLs specified in `knowledge_files.txt`
- Creates `knowledge/` directory if it doesn't exist
- Only downloads missing files to avoid unnecessary requests
- Provides feedback on download success/failure

**Knowledge Files Configuration (`knowledge_files.txt`)**
- Maps expected filenames to download URLs
- Format: `filename|download_url`
- Supports comments and blank lines for organization
- Easily updatable when URLs change

**Main Application (`app.py`)**
- Streamlit web interface with chat functionality
- Uses `ConversationalRetrievalChain` for context-aware conversations
- Automatic vector store creation from PDF and TXT documents
- Session state management for conversation memory
- Source citation for retrieved documents
- Automatic knowledge file downloading when missing

**Vector Store System**
- Uses FAISS for vector storage with `OllamaEmbeddings` (nomic-embed-text:latest)
- Automatic creation from PDF and TXT files in `knowledge/` directory
- Chunking: 1000 characters with 200 overlap
- Stored in `vectorstore/faiss_index/`

**Knowledge Base**
- Located in `knowledge/` directory (gitignored)
- Downloads therapy resources from URLs defined in `knowledge_files.txt`
- Supports both PDF and TXT files
- Automatically downloads missing files on first run or when "Rebuild Knowledge Base" is clicked

**LLM Integration**
- Uses `ChatOllama` with llama3.2:latest model
- Local inference via Ollama (no external API costs)
- Temperature: 0.7 for balanced responses

### Key Configuration

**Environment Variables (`.env`)**
```
KNOWLEDGE_PATH=./knowledge              # Directory containing therapy PDFs
VECTORSTORE_PATH=./vectorstore         # Directory for FAISS vector storage
OLLAMA_HOST=http://localhost:11434     # Ollama server URL
OLLAMA_MODEL=llama3.2:latest          # Primary chat model
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest  # Embedding model
```

### Data Flow

1. **Initial Load**: App checks for existing vector store, creates from PDFs if missing
2. **User Query**: Input processed through `ConversationalRetrievalChain`
3. **Retrieval**: Similarity search against vector store returns relevant text chunks
4. **Generation**: LLM generates response using retrieved context + conversation history
5. **Response**: Displayed with source citations and added to conversation memory

### Critical Implementation Details

**Error Handling**
- Comprehensive try/catch blocks around all external service calls
- Graceful fallbacks when vector store loading fails
- User-friendly error messages in Streamlit UI

**Performance Considerations**
- Vector store persists between sessions to avoid reprocessing PDFs
- Lazy loading of models and embeddings
- Progress indicators for long-running operations

**Memory Management**
- `ConversationBufferMemory` maintains chat context
- Session state preserves user interactions
- Vector store cached to avoid rebuilds

**Therapy-Specific Features**
- Evidence-based responses grounded in provided materials
- Source transparency for professional accountability
- Confidential local processing (no cloud data transmission)

## Development Notes

**Environment Dependencies**
- Ollama must be running locally on port 11434
- Required models: `llama3.2:latest`, `nomic-embed-text:latest`
- Vectorstore directory is gitignored to prevent large commits
- Knowledge directory contents are gitignored and downloaded dynamically

**Code Style**
- Follows existing patterns from `AGENT.md`
- Python 3.x conventions with snake_case
- Environment variables for configuration via python-dotenv
- Type hints not present but acceptable to add
- F-strings for string formatting
- Comprehensive docstrings for public functions

**Testing Approach**
- No formal test suite - test through application interaction
- `test_ollama.py` verifies core Ollama connectivity
- Manual testing via conversation flows with therapy content