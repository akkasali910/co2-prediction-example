import streamlit as st
import os
import sys
import requests
import json

# --- LlamaIndex Imports ---
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# --- Disable LlamaIndex Telemetry ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
# --- Disable Tokenizers Parallelism to avoid warnings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration (Max Speed Priority) ---
OLLAMA_MODEL = "tinyllama"
OLLAMA_REQUEST_TIMEOUT = 300.0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MCP_SERVER_URL = "http://127.0.0.1:5001/mcp/v1/get_model_context"
# ------------------------------------------

@st.cache_resource(show_spinner="Initializing LLM...")
def setup_llm():
    """
    Initializes and caches the local Ollama LLM.
    This function will only run once per session.
    """
    st.info(f"Initializing Ollama model: **{OLLAMA_MODEL}**")
    try:
        # Initialize LLM (Ollama)
        llm = Ollama(
            model=OLLAMA_MODEL,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
            max_tokens=512, # Limit response length to prevent rambling
            stop_sequences=["\nUser:", "\n\n"], # Stop if it tries to start a new turn or repeats
            temperature=0.1 # Make the output more deterministic and less prone to repetition
        )
        llm.complete("Hi", max_tokens=1)  # Test call
        st.success(f"✅ Ollama LLM '{OLLAMA_MODEL}' is ready!")
        return llm
    except Exception as e:
        st.error(f"Ollama Connection Error: Could not connect to the '{OLLAMA_MODEL}' model. Please ensure Ollama is running and the model is pulled.")
        return None

def main():
    """Main Streamlit application logic."""
    st.set_page_config(page_title="Directory Analyzer", layout="wide")

    st.title("📄 Directory Content Analyzer (Client Mode)")
    st.markdown(f"Powered by LlamaIndex, Ollama (`{OLLAMA_MODEL}`), and Streamlit.")

    # --- Sidebar for status and instructions ---
    with st.sidebar:
        st.header("Setup Instructions")
        st.info(
            "This application acts as a client. Before using it, please ensure:\n\n"
            "1. **The MCP Server is running**.\n   - Run `python local_mcp_server.py --port 5001` in a separate terminal.\n\n"
            "2. **The Vector Index is built**.\n   - Run `python build_index.py` if you haven't already.\n\n"
            "3. **Ollama is running** with the `{OLLAMA_MODEL}` model."
        )

    # --- Setup RAG Engine ---
    # This will run on the first load and the result is cached.
    llm = setup_llm()

    st.divider()

    if llm is None:
        st.warning("LLM not initialized. Please check the Ollama connection error above.")
        return

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to analyze the files in this directory. What would you like to know?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask a question about the files...", disabled=(llm is None)):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                # 1. Retrieve context from the MCP server
                with st.spinner("Retrieving context from MCP server..."):
                    headers = {'Content-Type': 'application/json'}
                    payload = json.dumps({"user_text": prompt})
                    response = requests.post(MCP_SERVER_URL, headers=headers, data=payload, timeout=60)
                    response.raise_for_status()
                    
                    context_items = response.json().get("context_items", [])
                    context_string = "\n\n".join([item['content'] for item in context_items])

                # 2. Build a new prompt with the retrieved context
                final_prompt = (
                    "You are an expert file analyzer. Use the following context to answer the user's query. "
                    "Only use the information from the provided context. If the answer is not in the context, "
                    "state that you cannot find the answer in the provided files.\n\n"
                    "--- CONTEXT ---\n"
                    f"{context_string}\n"
                    "--- END CONTEXT ---\n\n"
                    f"User Query: {prompt}"
                )

                # 3. Stream the final answer from the local LLM
                response_stream = llm.stream_complete(final_prompt)

                for token in response_stream:
                    full_response += token.delta
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)

            except requests.exceptions.RequestException as e:
                error_message = f"Could not connect to MCP server at `{MCP_SERVER_URL}`. Please ensure it's running. Details: {e}"
                st.error(error_message)
                full_response = error_message
            except Exception as e:
                error_message = f"An error occurred during query: {e}"
                st.error(error_message)
                full_response = error_message

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()
