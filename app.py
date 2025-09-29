import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

def load_vectorstore():
    """Load or create the vector store with Ollama embeddings."""
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    knowledge_path = os.getenv("KNOWLEDGE_PATH", "./knowledge")

    try:
        # Try to load existing vector store first
        save_path = os.path.join(vectorstore_path, "faiss_index")
        if os.path.exists(save_path):
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            st.success("Loaded existing vector store")
            return vectorstore
        else:
            # Create vector store if it doesn't exist
            return create_vectorstore(knowledge_path, vectorstore_path)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def download_knowledge_files():
    """Download knowledge files from URLs if they don't exist."""
    try:
        from download_knowledge import load_knowledge_files_config, download_file
        from pathlib import Path

        project_root = Path(__file__).parent
        knowledge_dir = project_root / "knowledge"
        knowledge_dir.mkdir(exist_ok=True)

        files_config = load_knowledge_files_config()

        missing_files = []
        for filename, url in files_config:
            file_path = knowledge_dir / filename
            if not file_path.exists():
                missing_files.append((filename, url))

        if missing_files:
            with st.spinner(f"Downloading {len(missing_files)} knowledge files..."):
                success_count = 0
                for filename, url in missing_files:
                    if download_file(filename, url, knowledge_dir):
                        success_count += 1

                if success_count == len(missing_files):
                    st.success(f"Successfully downloaded {success_count} knowledge files")
                else:
                    st.warning(f"Downloaded {success_count}/{len(missing_files)} files. Some resources may be unavailable.")

        return True
    except Exception as e:
        st.error(f"Error downloading knowledge files: {str(e)}")
        return False

def create_vectorstore(knowledge_path, vectorstore_path):
    """Create a new vector store from PDF files."""
    if not os.path.exists(knowledge_path):
        st.error(f"Knowledge directory '{knowledge_path}' does not exist.")
        return None

    # Try to download knowledge files if needed
    pdf_files = [f for f in os.listdir(knowledge_path) if f.endswith('.pdf')]
    txt_files = [f for f in os.listdir(knowledge_path) if f.endswith('.txt')]

    if not pdf_files and not txt_files:
        if download_knowledge_files():
            # Check again after downloading
            pdf_files = [f for f in os.listdir(knowledge_path) if f.endswith('.pdf')]
            txt_files = [f for f in os.listdir(knowledge_path) if f.endswith('.txt')]

    if not pdf_files and not txt_files:
        st.error(f"No knowledge files found in '{knowledge_path}' directory after attempting download.")
        return None

    with st.spinner("Processing knowledge documents..."):
        try:
            all_documents = []

            # Process PDF files
            for pdf_file in pdf_files:
                pdf_path = os.path.join(knowledge_path, pdf_file)
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # Add metadata
                for doc in documents:
                    doc.metadata['source'] = pdf_file

                all_documents.extend(documents)

            # Process TXT files
            for txt_file in txt_files:
                txt_path = os.path.join(knowledge_path, txt_file)
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(txt_path, encoding='utf-8')
                documents = loader.load()

                # Add metadata
                for doc in documents:
                    doc.metadata['source'] = txt_file

                all_documents.extend(documents)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(all_documents)

            # Create embeddings with Ollama
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Save vector store
            os.makedirs(vectorstore_path, exist_ok=True)
            save_path = os.path.join(vectorstore_path, "faiss_index")
            vectorstore.save_local(save_path)

            st.success(f"Created vector store with {len(chunks)} chunks from {len(pdf_files + txt_files)} files")
            return vectorstore

        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

def create_conversation_chain(vectorstore):
    """Create conversation chain with Ollama LLM."""
    try:
        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.7,
            streaming=True
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )

        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def get_ai_response(conversation_chain, question):
    """Get response from AI therapist."""
    try:
        if conversation_chain is None:
            return "I'm sorry, but I'm not properly initialized. Please try refreshing the page.", []

        response = conversation_chain({"question": question})
        answer = response.get("answer", "I apologize, but I couldn't generate a response.")
        source_documents = response.get("source_documents", [])

        return answer, source_documents
    except Exception as e:
        error_msg = f"I encountered an error: {str(e)}"
        return error_msg, []

def main():
    """Main application function."""
    st.set_page_config(
        page_title="AI Therapist",
        page_icon="🧠",
        layout="centered"
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS for better styling
    st.markdown("""
        <style>
            .user-message {
                background-color: #e3f2fd;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
            }
            .assistant-message {
                background-color: #f3e5f5;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
            }
            .source-info {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 AI Therapist")
    st.markdown("*Your compassionate mental health assistant*")

    # Initialize or load vector store
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = load_vectorstore()

        if st.session_state.vectorstore is not None:
            st.session_state.conversation_chain = create_conversation_chain(st.session_state.vectorstore)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}.** {source}")

    # Chat input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source_docs = get_ai_response(st.session_state.conversation_chain, prompt)

                # Format sources
                sources = []
                for doc in source_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    if source not in sources:
                        sources.append(source)

                # Display response
                st.markdown(answer)

                # Show sources if available
                if sources:
                    with st.expander("Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}.** {source}")

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI therapist provides mental health support based on therapeutic knowledge and techniques.

        **Features:**
        - Evidence-based responses
        - Confidential conversations
        - Access to therapeutic resources
        - 24/7 availability

        **Disclaimer:** This is not a replacement for professional mental health care.
        """)

        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Rebuild Knowledge Base"):
            st.session_state.vectorstore = None
            st.session_state.conversation_chain = None
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()