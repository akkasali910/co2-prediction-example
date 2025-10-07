# Architecture Diagram: Local RAG with Streamlit

This diagram illustrates the architecture of the `log_analyzer_app.py` Streamlit application. It shows the flow for both indexing local files and answering user queries using a local RAG (Retrieval-Augmented Generation) pipeline.

```mermaid
graph LR
    subgraph "User Interface (Streamlit)"
        direction LR
        UI_Sidebar[<br><b>Sidebar Controls</b><br>Load Directory<br>Upload Files]
        UI_Chat[<br><b>Chat Interface</b><br>Ask Questions<br>View Responses]
    end

    subgraph "Local File System"
        direction LR
        FS[Project Directory]
        Uploads[Uploaded Files]
    end

    subgraph "RAG Engine (Cached in Memory via @st.cache_resource)"
        direction TB
        Setup["setup_rag_engine()"]
        Update["update_index_with_uploads()"]
        
        subgraph "Core Components"
            direction LR
            Loader(SimpleDirectoryReader)
            Embed(HuggingFace<br>Embedding Model)
            LLM(Ollama<br>'tinyllama')
            VS(ChromaVectorStore<br><i>In-Memory</i>)
            QE(Query Engine)
        end
    end

    User([👤 User]) --> UI_Sidebar
    User --> UI_Chat

    UI_Sidebar -- "1a. Click 'Load & Index'" --> Setup
    Setup -- "Reads files from" --> FS
    Setup -- "Builds Index in" --> VS
    Setup -- "Creates" --> QE

    UI_Sidebar -- "1b. Upload & Click 'Add Files'" --> Update
    Update -- "Reads files from" --> Uploads
    Update -- "Inserts new nodes into" --> VS

    UI_Chat -- "2. Asks a question" --> QE
    QE -- "3. Retrieves context from" --> VS
    QE -- "4. Augments prompt and sends to" --> LLM
    LLM -- "5. Streams response back to" --> QE
    QE -- "6. Displays response in" --> UI_Chat
```

---

### Diagram Explanation

1.  **User Interface (Streamlit)**: This is the web front-end. The user interacts with the **Sidebar Controls** to load data and the **Chat Interface** to ask questions.

2.  **Local File System**: Represents the files on the machine where the app is running. The `setup_rag_engine()` function reads from the `Project Directory`, while `update_index_with_uploads()` reads from temporary `Uploaded Files`.

3.  **RAG Engine**: This is the core logic of the application, cached in memory by Streamlit to avoid re-initialization on every interaction.
    *   **Core Components**:
        *   `SimpleDirectoryReader`: Loads and parses text from files.
        *   `HuggingFace Embedding Model`: Converts text chunks into vector embeddings.
        *   `Ollama 'tinyllama'`: The local Large Language Model that generates answers.
        *   `ChromaVectorStore`: The in-memory vector database that stores and searches for embeddings.
        *   `Query Engine`: Orchestrates the query process (embedding the query, retrieving context, and calling the LLM).

4.  **Flows**:
    *   **Indexing (1a & 1b)**: The user initiates indexing via the sidebar. The RAG engine reads files, creates embeddings, and stores them in the Chroma vector store.
    *   **Querying (2-6)**: The user asks a question in the chat. The Query Engine retrieves relevant context from the vector store, combines it with the user's question, and sends it to the Ollama LLM to generate a final, streamed answer.