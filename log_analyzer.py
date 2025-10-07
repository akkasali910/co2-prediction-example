import os
import sys
import requests
import json

# --- LlamaIndex Imports ---
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# --- Disable Telemetry/Warnings ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
OLLAMA_MODEL = "tinyllama" 
OLLAMA_REQUEST_TIMEOUT = 300.0 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MCP_SERVER_URL = "http://127.0.0.1:5001/mcp/v1/get_model_context"
# ------------------------------------------

def setup_llm():
    """Initializes the local Ollama LLM."""
    
    try:
        print("\n--- 1. Initializing Ollama (tinyllama) ---")
        # Initialize LLM (Ollama)
        llm = Ollama(
            model=OLLAMA_MODEL,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
            max_tokens=512, # Limit response length to prevent rambling
            stop_sequences=["\nUser:", "\n\n"], # Stop if it tries to start a new turn or repeats
            temperature=0.1 # Make the output more deterministic and less prone to repetition
        )
        llm.complete("Hi", max_tokens=1) # Test call
        print(f"✅ Ollama LLM '{OLLAMA_MODEL}' initialized successfully.")
        return llm
    except Exception as e:
        print(f"❌ Ollama ERROR: Model '{OLLAMA_MODEL}' not accessible or server not running.")
        print(f"   Action: Ensure 'ollama serve' is running and you have pulled the model using: 'ollama pull {OLLAMA_MODEL}'")
        sys.exit(1)

def analyze_console():
    """Main console loop for querying the MCP server and generating answers."""
    
    llm = setup_llm()
    
    print("\n=======================================================")
    print("      Directory Analyzer is READY (Client Mode)")
    print("=======================================================")
    print(f"   - Ensure the MCP server is running.")
    print("   Enter your query or type 'exit' or 'quit' to end.")
    print("=======================================================\n")
    
    while True:
        try:
            user_query = input("❓ Your Query: ")
            
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Exiting analyzer. Goodbye!")
                break
            
            if not user_query.strip():
                continue

            print("\n🤖 AI Response (Streaming):")

            # 1. Retrieve context from the MCP server
            try:
                print("   (Retrieving context from MCP server...)", end="\r")
                headers = {'Content-Type': 'application/json'}
                payload = json.dumps({"user_text": user_query})
                response = requests.post(MCP_SERVER_URL, headers=headers, data=payload)
                response.raise_for_status()
                
                context_items = response.json().get("context_items", [])
                context_string = "\n\n".join([item['content'] for item in context_items])
                print("                                           ", end="\r") # Clear the line
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Error: Could not connect to MCP server at {MCP_SERVER_URL}.")
                print(f"   Please ensure 'local_mcp_server.py' is running. Details: {e}")
                continue
            
            # 2. Build a new prompt with the retrieved context
            final_prompt = (
                "You are an expert file analyzer. Use the following context to answer the user's query. "
                "Only use the information from the provided context. If the answer is not in the context, "
                "state that you cannot find the answer in the provided files.\n\n"
                "--- CONTEXT ---\n"
                f"{context_string}\n"
                "--- END CONTEXT ---\n\n"
                f"User Query: {user_query}"
            )

            # 3. Stream the final answer from the local LLM
            response_stream = llm.stream_complete(final_prompt)

            # Iterate over the tokens in the response stream
            for token in response_stream:
                print(token.delta, end="", flush=True)

            print("\n-------------------------------------------------------\n")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting analyzer. Goodbye!")
            break
        except Exception as e:
            # We catch specific PyTorch errors related to memory/device to give better feedback
            if "CUDA out of memory" in str(e):
                 print("\n❌ Error: CUDA out of memory. Try closing other GPU applications or reducing context size.")
            else:
                print(f"\n❌ Error during query execution: {e}")
                print("   Please ensure your Ollama server is running and accessible.")

if __name__ == '__main__':
    analyze_console()
