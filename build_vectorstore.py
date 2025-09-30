import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def build_vectorstore():
    """Builds a FAISS vector store from PDF documents.

    This function scans a specified directory for PDF files, loads their
    content, splits the text into manageable chunks, and then uses OpenAI
    embeddings to create a vector representation of the text. The resulting
    vector store is saved to a specified directory.

    The paths for the knowledge base and the vector store are configured
    via environment variables `KNOWLEDGE_PATH` and `VECTORSTORE_PATH`.

    The process includes:
    1.  Reading `KNOWLEDGE_PATH` and `VECTORSTORE_PATH` from environment
        variables.
    2.  Scanning the knowledge directory for PDF files.
    3.  Loading and processing each PDF, extracting text and metadata.
    4.  Splitting the documents into chunks using RecursiveCharacterTextSplitter.
    5.  Generating embeddings for the chunks using OpenAIEmbeddings.
    6.  Creating and saving a FAISS vector store to the specified path.
    7.  Running a quick test query to verify the vector store's integrity.
    """
    
    # Get paths from environment variables
    knowledge_path = os.getenv("KNOWLEDGE_PATH", "./knowledge")
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    
    # Check if knowledge directory exists
    if not os.path.exists(knowledge_path):
        print(f"Error: Knowledge directory '{knowledge_path}' does not exist.")
        return
    
    # Get all PDF files in knowledge directory
    pdf_files = [f for f in os.listdir(knowledge_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{knowledge_path}' directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    all_documents = []
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(knowledge_path, pdf_file)
        print(f"\nProcessing {pdf_file}...")
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"  Loaded {len(documents)} pages")
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = pdf_file
            
            all_documents.extend(documents)
            
        except Exception as e:
            print(f"  Error processing {pdf_file}: {str(e)}")
            continue
    
    if not all_documents:
        print("No documents were successfully processed.")
        return
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} text chunks")
    
    # Create embeddings
    print("Creating embeddings...")
    try:
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        print("Building vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Create vectorstore directory if it doesn't exist
        os.makedirs(vectorstore_path, exist_ok=True)
        
        # Save vector store
        save_path = os.path.join(vectorstore_path, "faiss_index")
        vectorstore.save_local(save_path)
        print(f"Vector store saved to: {save_path}")
        
        # Test retrieval
        print("\nTesting vector store retrieval...")
        test_query = "What is anxiety?"
        results = vectorstore.similarity_search(test_query, k=2)
        
        print(f"Retrieved {len(results)} documents for test query:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"     Content preview: {doc.page_content[:100]}...")
        
        print("\nVector store built successfully!")
        
    except Exception as e:
        print(f"Error creating embeddings or vector store: {str(e)}")
        print("Please check your OpenAI API key and internet connection.")

if __name__ == "__main__":
    start_time = time.time()
    build_vectorstore()
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
