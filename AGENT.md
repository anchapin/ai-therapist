# AI Therapist Project - Agent Instructions

## Build/Lint/Test Commands
- `python build_vectorstore.py` - Build vector store from PDFs in knowledge/ directory
- `python -m pip install -r requirements.txt` - Install dependencies
- No formal test suite - test by running build_vectorstore.py and checking output

## Architecture & Structure
- **Main script**: `build_vectorstore.py` - LangChain-based vectorstore builder
- **Knowledge base**: `knowledge/` - PDF documents (therapy materials, anxiety resources)
- **Vector store**: `vectorstore/` - FAISS embeddings generated from PDFs
- **Environment**: `ai-therapist-env/` - Python virtual environment
- **Dependencies**: LangChain, OpenAI embeddings, FAISS, PyPDF

## Code Style Guidelines
- Python 3.x standard conventions (snake_case, docstrings)
- Use environment variables for paths and API keys (load via python-dotenv)
- Error handling with try/except blocks and informative messages
- F-strings for string formatting
- Type hints not used but acceptable to add
- Keep chunk_size=1000, chunk_overlap=200 for text splitting
- Use os.path.join() for cross-platform path handling
- Print progress messages for long-running operations
- Metadata preservation when processing documents
