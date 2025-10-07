# Local MCP Server Documentation

This document explains how to use the `local_mcp_server.py` script. This script runs a local web server that exposes your vectorized file index as a knowledge source, following the **Model Context Protocol (MCP)**. This allows AI agents, like those built with the AWS Strands SDK, to retrieve relevant context from your local files to answer questions.

## 1. Purpose

The primary purpose of this server is to act as a bridge between an AI agent and your local data.

-   It loads a persistent vector index created by `build_index.py`.
-   It provides a standardized API endpoint (`/mcp/v1/get_model_context`) for retrieving context.
-   When an agent sends a query (e.g., a user's question), the server performs a vector search on the index and returns the most relevant text chunks from your files.

## 2. Prerequisites

Before running the server, ensure you have completed the following:

1.  **Installed Dependencies**: You must have Flask and all LlamaIndex-related libraries installed.
    ```bash
    pip install Flask llama-index chromadb sentence-transformers
    ```

2.  **Built the Vector Index**: You must run the `build_index.py` script first. This creates the `./chroma_db_index` directory that the server needs to load the knowledge base.
    ```bash
    python build_index.py
    ```

## 3. How to Run the Server

Once the prerequisites are met, you can start the server with a simple command:

```bash
python local_mcp_server.py
```

You will see output indicating that the server is running and listening for requests, typically on `http://127.0.0.1:5000`.

```
🚀 Starting Local MCP Server...
   - Make sure you have run 'pip install Flask'.
   - Ensure the index exists at './chroma_db_index' by running 'build_index.py'.
   - Listening on http://127.0.0.1:5000/mcp/v1/get_model_context
```

## 4. API Endpoint Details

-   **URL**: `http://127.0.0.1:5000/mcp/v1/get_model_context`
-   **Method**: `POST`
-   **Content-Type**: `application/json`

### Request Body

The server accepts a JSON payload with either a single query or a list of queries.

**Single Query:**
```json
{
  "user_text": "your question here"
}
```

**Multiple Queries:**
```json
{
  "user_texts": [
    "first question",
    "second question"
  ]
}
```

### Success Response (200 OK)

The server returns a JSON object containing a list of `context_items`. Each item is a text chunk retrieved from the vector index.

```json
{
  "context_items": [
    { "item_type": "text", "content": "Relevant text chunk from a file..." },
    { "item_type": "text", "content": "Another relevant text chunk..." }
  ]
}
```

## 5. Testing with `curl`

You can easily test the running server from your terminal using `curl`.

### Test with a Single Query
```bash
curl --request POST \
  --url http://127.0.0.1:5000/mcp/v1/get_model_context \
  --header 'Content-Type: application/json' \
  --data '{"user_text": "summarize the purpose of the python scripts"}'
```

### Test with Multiple Queries
```bash
curl --request POST \
  --url http://127.0.0.1:5000/mcp/v1/get_model_context \
  --header 'Content-Type: application/json' \
  --data '{
    "user_texts": [
      "what is the purpose of the cleanup script",
      "how is the sagemaker pipeline deployed"
    ]
  }'
```