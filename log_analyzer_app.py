import streamlit as st
import os
import sys
import re
# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Disable LlamaIndex Telemetry ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
# --- Disable Tokenizers Parallelism to avoid warnings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration (Max Speed Priority) ---
from guardrails import Guardrails
from config import (OLLAMA_MODEL, OLLAMA_REQUEST_TIMEOUT, EMBEDDING_MODEL_NAME, 
                    CHROMA_PERSIST_DIR, TOP_K_CHUNKS)
# ------------------------------------------

@st.cache_resource(show_spinner="Initializing RAG Engine...")
def setup_chat_engine():
    """
    Initializes and caches the LlamaIndex chat engine.
    This function will only run once per session.
    """
    st.info(f"1. Initializing Ollama model: **{OLLAMA_MODEL}**")
    try:
        llm = Ollama(
            model=OLLAMA_MODEL,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
            max_tokens=512,
            stop_sequences=["\nUser:", "\n\n"],
            temperature=0.1
        )
        llm.complete("Hi", max_tokens=1)  # Test call
        st.info(f"2. Initializing Retriever from local index...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.llm = llm

        db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_collection = db.get_or_create_collection("directory_analysis_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        retriever = index.as_retriever(similarity_top_k=TOP_K_CHUNKS)

        st.info(f"3. Initializing Chat Engine...")
        system_prompt = (
            "You are an expert file analyzer. Your task is to answer the user's questions based on the provided context. "
            "The context is retrieved from a local directory of files. "
            "Always be helpful and answer concisely. If the answer is not in the context, "
            "state that you cannot find the answer in the provided files. "
            "You have access to the conversation history."
        )
        chat_engine = ContextChatEngine.from_defaults(retriever=retriever, llm=llm, system_prompt=system_prompt)
        st.success("✅ RAG Engine is ready!")
        return chat_engine

    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.warning("Please ensure 'ollama serve' is running and you have run 'python build_index.py' first.")
        return None

def main():
    """Main Streamlit application logic."""
    st.set_page_config(page_title="Directory Analyzer", layout="wide")

    st.title("📄 Directory Content Analyzer (Conversational Mode)")
    st.markdown(f"Powered by LlamaIndex, Ollama (`{OLLAMA_MODEL}`), and Streamlit.")

    # --- Sidebar for status and instructions ---
    with st.sidebar:
        st.header("Setup Instructions")
        st.info(
            "This application runs in local mode. Please ensure:\n\n"
            "1. **The Vector Index is built**.\n   - Run `python build_index.py` if you haven't already.\n\n"
            "2. **Ollama is running** with the `{OLLAMA_MODEL}` model."
        )

        st.divider()
        st.header("Chat Controls")
        if st.button("Clear Chat History"):
            # Reset the chat engine's internal state
            if chat_engine:
                chat_engine.reset()
            # Clear the UI message history and the engine's history stored in session state
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to answer questions about the files in this directory."}]
            if 'chat_engine_history' in st.session_state:
                del st.session_state.chat_engine_history
            
            st.success("Chat history cleared!")
            # Rerun the app to reflect the cleared state
            st.rerun()

    # --- Setup RAG Engine ---
    # This will run on the first load and the result is cached.
    chat_engine = setup_chat_engine()
    guardrails = Guardrails()

    st.divider()

    if chat_engine is None:
        st.warning("Application is not ready. Please check the errors above.")
        return

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to answer questions about the files in this directory."}]
        # Initialize the chat engine's history in session state
        st.session_state.chat_engine_history = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask a question about the files...", disabled=(chat_engine is None)):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Input Guardrail Check ---
        is_safe, message = guardrails.check_input(prompt)
        if not is_safe:
            with st.chat_message("assistant"):
                st.error(f"GUARDRAIL: {message}")
            st.session_state.messages.append({"role": "assistant", "content": f"GUARDRAIL: {message}"})
            return

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Thinking..."):
                    # Use the chat engine to get a streaming response
                    streaming_response = chat_engine.stream_chat(prompt, chat_history=st.session_state.chat_engine_history)

                # --- Output Guardrail Processing ---
                for token in guardrails.stream_and_sanitize(streaming_response):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                # Update the chat engine's history
                st.session_state.chat_engine_history = chat_engine.chat_history

            except Exception as e:
                error_message = f"An error occurred during query: {e}"
                st.error(error_message)
                full_response = error_message

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()
