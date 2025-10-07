# Project Troubleshooting and Fixes

This document outlines the key technical challenges encountered during the development of the RAG-based directory analyzer application and the solutions implemented to resolve them. This serves as a "lessons learned" guide for the project.

---

## 1. Decoupling the Architecture: From Monolith to Client-Server

### Initial Problem
The first versions of `log_analyzer.py` and `log_analyzer_app.py` were monolithic. They handled data loading, indexing, and querying all in the same process. This was slow to start up and not scalable, as the entire index had to be rebuilt in memory each time.

### Solution: Separation of Concerns
The architecture was refactored into three distinct, more robust components:

1.  **`build_index.py`**: A dedicated, one-time script to read local files and create a **persistent vector store** on disk using ChromaDB. This separates the slow, heavy indexing task from the main application.

2.  **`local_mcp_server.py`**: A lightweight Flask server that loads the pre-built persistent index. Its sole job is to expose a retrieval API endpoint (`/mcp/v1/get_model_context`) that accepts a query and returns relevant context chunks. This is the "retrieval" part of RAG.

3.  **Client Applications (`log_analyzer.py`, `log_analyzer_app.py`)**: These were refactored to be pure clients. They no longer load data or build indexes. Instead, they:
    *   Accept a user query.
    *   Make an HTTP request to the MCP server to get context.
    *   Use a local Ollama LLM to synthesize an answer based on that context.

This client-server model is more scalable, faster to start, and aligns with modern MLOps practices.

---

## 2. The Garbled and Repetitive Streaming Response

### Problem
The most significant and frustrating issue was the response from the language model being streamed incorrectly, resulting in garbled, "stuttering" text like this:

```
TheThe machineThe machine learningThe machine learning life...
```

### Root Cause
The client applications were handling the streaming response objects from LlamaIndex incorrectly. The code was using `token.text` inside the streaming loop. In LlamaIndex's streaming API, `token.text` contains the *entire accumulated response so far*, not just the new token. The loop was therefore re-printing the entire response with each new word added, causing the garbled output.

### The Fix: Using `.delta`
The solution was to change the streaming loop to use `token.delta`. The `.delta` attribute contains only the **new chunk of text** for that specific step in the stream.

**Incorrect Code:**
```python
# This caused the garbled output
for token in response_stream:
    full_response += token.text # WRONG: token.text is the full accumulated string
    response_placeholder.markdown(full_response + "▌")
```

**Corrected Code:**
```python
# This provides a smooth, clean stream
for token in response_stream:
    full_response += token.delta # CORRECT: token.delta is only the new text
    response_placeholder.markdown(full_response + "▌")
```
This simple change completely resolved the streaming issue and resulted in a smooth, typewriter-style effect as intended.

---

## 3. Mitigating Model Hallucination and Repetition

### Problem
Even with correct streaming, the small `tinyllama` model would sometimes get stuck in a loop, repeating phrases or sentences.

### Root Cause
Smaller models often have a weaker sense of when to stop generating text. This can be due to limited context or a tendency to fall into repetitive generation patterns.

### Solution: Adding Guardrails to the LLM
We configured the `Ollama` instance with specific parameters to guide its behavior and prevent it from rambling:

```python
llm = Ollama(
    model=OLLAMA_MODEL,
    max_tokens=512,
    stop_sequences=["\nUser:", "\n\n"],
    temperature=0.1
)
```

*   **`max_tokens=512`**: A hard limit on the response length.
*   **`stop_sequences`**: Explicitly tells the model to stop if it tries to generate a new user prompt or multiple newlines, which are common precursors to repetition.
*   **`temperature=0.1`**: Makes the model's output more deterministic and less random, which helps it stay on topic and avoid repetitive loops.

---

## 4. Minor Issues and Fixes

*   **`AttributeError: 'RetrieverQueryEngine' object has no attribute 'stream_query'`**: The `stream_query` method was deprecated in LlamaIndex. This was fixed by using the standard `.query()` method and iterating over the `response.response_gen` attribute.
*   **`Address already in use`**: The MCP server could not start on port 5000 because it was occupied by the macOS AirPlay Receiver. This was fixed by adding a `--port` command-line argument to `local_mcp_server.py`, allowing it to run on a different port (e.g., 5001).
*   **`AttributeError: 'RetrieverQueryEngine' object has no attribute 'index'`**: When trying to add uploaded files to the index, the code tried to access `.index` directly from the query engine. This was fixed by accessing it through the correct path: `query_engine.retriever.index`.
*   **Python Errors (`IndentationError`, `UnboundLocalError`)**: Standard Python syntax and scope errors were identified and corrected as they appeared.