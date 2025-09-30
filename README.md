# 🧠 AI Therapist

The AI Therapist is a conversational AI application designed to provide compassionate and supportive mental health assistance. It leverages a Retrieval-Augmented Generation (RAG) architecture, using local language models via Ollama and a curated knowledge base of therapeutic materials. The application is built with Streamlit, LangChain, and FAISS.

## Features

- **Local & Private**: Runs entirely on your local machine using Ollama, ensuring your conversations remain private and confidential.
- **Evidence-Based**: Responses are grounded in a knowledge base of therapeutic documents (e.g., articles on anxiety, CBT worksheets).
- **Conversational Memory**: Remembers the context of your conversation for a more natural and coherent interaction.
- **Source-Citing**: Can cite the source documents it used to generate a response.
- **Easy to Use**: Simple, intuitive chat interface powered by Streamlit.

## Architecture Overview

The project is composed of several key components:

- **`app.py`**: The main Streamlit application. It handles the user interface, chat logic, and on-the-fly creation of the vector store if it doesn't exist.
- **`download_knowledge.py`**: A utility script to download the PDF and TXT files that form the knowledge base. It reads a list of URLs from `knowledge_files.txt`.
- **`build_vectorstore.py`**: A script to manually build the vector store using OpenAI embeddings. This is an alternative to the in-app process, which uses local Ollama embeddings.
- **`test_ollama.py`**: A simple script to verify that the connection to the Ollama server is working correctly for both embeddings and chat models.
- **`knowledge/`**: This directory stores the source documents (PDFs, TXTs) that the AI uses as its knowledge base.
- **`vectorstore/`**: This directory holds the FAISS index, which is the vectorized representation of the knowledge base.
- **`.env`**: The environment file for configuring paths and API keys.

## Setup and Installation

### 1. Prerequisites

- **Python**: Version 3.8 or higher.
- **Ollama**: You must have [Ollama](https://ollama.com/) installed and running.
- **Ollama Models**: Pull the required models by running:
  ```bash
  ollama pull llama3.2:latest
  ollama pull nomic-embed-text:latest
  ```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies

Install all the required Python packages:

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file by copying the template:

```bash
cp template.env .env
```

Review the `.env` file. The default paths are usually sufficient:
- `KNOWLEDGE_PATH`: Path to the directory with knowledge files (default: `./knowledge`).
- `VECTORSTORE_PATH`: Path to the directory where the FAISS index will be stored (default: `./vectorstore`).
- `OPENAI_API_KEY`: Required only if you intend to use the `build_vectorstore.py` script.

### 6. Prepare the Knowledge Base

The application will automatically try to download knowledge files on first run. However, you can also do this manually beforehand.

```bash
python download_knowledge.py
```
This script reads `knowledge_files.txt` and downloads the specified documents into the `knowledge/` directory. You can also add your own PDF or TXT files to this directory.

## Usage

### Running the AI Therapist App

To start the main application, run the following command:

```bash
streamlit run app.py
```

The first time you run the app, it will:
1. Check for knowledge files and download them if missing.
2. Process the documents and create a new FAISS vector store using the Ollama `nomic-embed-text` model. This may take a few moments.
3. Subsequent launches will be much faster as the app will load the pre-existing vector store.

The interface includes a sidebar with options to **Clear Conversation** or **Rebuild Knowledge Base** from scratch.

### Using the Command-Line Scripts

- **Test Ollama Connection**:
  Before running the app, you can verify your Ollama setup:
  ```bash
  python test_ollama.py
  ```

- **Download Knowledge Files Manually**:
  ```bash
  python download_knowledge.py
  ```

- **Build Vector Store with OpenAI (Optional)**:
  If you prefer to use OpenAI embeddings, ensure your `OPENAI_API_KEY` is set in the `.env` file and run:
  ```bash
  python build_vectorstore.py
  ```
  Note: The main application `app.py` is hardcoded to use Ollama embeddings. This script is provided as an alternative for building the vector store.

## Disclaimer

This AI Therapist is an experimental application and is **not a substitute for professional medical advice, diagnosis, or treatment**. If you are experiencing mental health issues, please consult with a qualified healthcare professional.