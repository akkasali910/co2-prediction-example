import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import torch

# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Disable LlamaIndex Telemetry ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
# --- Disable Tokenizers Parallelism to avoid warnings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration (Max Speed Priority) ---
OLLAMA_MODEL = "tinyllama"
OLLAMA_REQUEST_TIMEOUT = 300.0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = [".log", ".txt", ".py", ".md"]
TOP_K_CHUNKS = 2 # Increase to get more context
# ------------------------------------------

@st.cache_resource(show_spinner="Initializing RAG Engine...")
def setup_rag_engine(directory_path="."):
    """
    Initializes and caches the LlamaIndex RAG pipeline from a directory.
    This function will only run once per session.
    """
    # Determine the device for PyTorch (used by HuggingFace Embeddings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Initializing embeddings on device: **{device.upper()}**")

    try:
        # 1. Initialize Embeddings (HuggingFace)
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device=device)
    except Exception as e:
        st.error(f"Error initializing HuggingFace Embeddings: {e}")
        return None

    st.info(f"Initializing Ollama model: **{OLLAMA_MODEL}**")
    try:
        # 2. Initialize LLM (Ollama)
        llm = Ollama(model=OLLAMA_MODEL, request_timeout=OLLAMA_REQUEST_TIMEOUT)
        llm.complete("Hi", max_tokens=1)  # Test call

        Settings.llm = llm
        Settings.embed_model = embed_model
    except Exception as e:
        st.error(f"Ollama Connection Error: Could not connect to the '{OLLAMA_MODEL}' model. Please ensure Ollama is running and the model is pulled.")
        return None

    st.info("Reading directory files...")
    try:
        # 3. Document Loading
        reader = SimpleDirectoryReader(input_dir=directory_path, required_exts=ALLOWED_EXTENSIONS, recursive=True)
        documents = reader.load_data()

        if not documents:
            st.warning(f"No files found in the current directory with extensions: {', '.join(ALLOWED_EXTENSIONS)}.")
            return None

        indexed_files = [Path(doc.metadata['file_path']).name for doc in documents]
        st.info(f"Found and loaded {len(indexed_files)} files.")

    except Exception as e:
        st.error(f"Error reading directory files: {e}")
        return None

    st.info("Building vector index in memory...")
    # 4. Vector Store Setup (IN-MEMORY ChromaDB)
    db_client = chromadb.Client()
    chroma_collection = db_client.get_or_create_collection("directory_analysis_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Build Index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # 6. Setup Query Engine
    system_prompt = (
        "You are an expert file analyzer. Your task is to analyze the provided file "
        "snippets and answer the user's question concisely. The file snippets are prefixed "
        "with their original filename for context. "
        "Only use the information from the file content provided in the context. "
        "If the answer is not in the context, state that you cannot find the answer in the provided files."
    )

    query_engine = index.as_query_engine(
        system_prompt=system_prompt,
        similarity_top_k=TOP_K_CHUNKS,
        streaming=True
    )

    st.success("✅ RAG Engine is ready!")
    return query_engine

def update_index_with_uploads(uploaded_files):
    """Processes uploaded files and adds them to the existing index."""
    if "query_engine" not in st.session_state or st.session_state.query_engine is None:
        st.error("RAG Engine not initialized. Please load a directory first.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
        
        if not saved_files:
            st.warning("No files were uploaded.")
            return

        st.info(f"Reading {len(saved_files)} new files...")
        reader = SimpleDirectoryReader(input_dir=temp_dir)
        new_documents = reader.load_data()

        st.info("Inserting new documents into the vector index...")
        # The index is accessed through the retriever property of the query engine
        index = st.session_state.query_engine.retriever.index
        index.insert_nodes(new_documents)
        st.success(f"Successfully added {len(new_documents)} documents to the index!")

def main():
    """Main Streamlit application logic."""
    st.set_page_config(page_title="Directory Analyzer", layout="wide")

    st.title("📄 Directory Content Analyzer")
    st.markdown(f"Powered by LlamaIndex, Ollama (`{OLLAMA_MODEL}`), and Streamlit.")

    # --- Sidebar for controls ---
    with st.sidebar:
        st.header("Controls")
        st.write("Load data from the local directory where this app is running.")
        if st.button("Load & Index Directory"):
            # Clear previous state and re-initialize
            if "query_engine" in st.session_state:
                del st.session_state["query_engine"]
            st.cache_resource.clear()
            st.session_state.query_engine = setup_rag_engine()
        
        st.divider()
        
        st.header("Upload Files")
        st.write("Upload additional files to add to the index.")
        uploaded_files = st.file_uploader(
            "Choose files", accept_multiple_files=True, type=ALLOWED_EXTENSIONS
        )
        if uploaded_files:
            if st.button("Add Files to Index"):
                update_index_with_uploads(uploaded_files)


    # --- Setup RAG Engine ---
    # This will run on the first load and the result is cached.
    query_engine = setup_rag_engine()

    st.divider()

    if query_engine is None:
        st.info("Please load a directory or upload files using the sidebar to begin.")
        return

    # --- Chat Interface ---
    if "messages" not in st.session_state or not query_engine:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to analyze the files in this directory. What would you like to know?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask a question about the files...", disabled=(query_engine is None)):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                # Use the cached query engine
                streaming_response = query_engine.query(prompt)

                # Stream the response to the UI
                for token in streaming_response.response_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)

            except Exception as e:
                error_message = f"An error occurred during query: {e}"
                st.error(error_message)
                full_response = error_message

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()
