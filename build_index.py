import os
import sys
from pathlib import Path

# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Local Imports ---
from config import EMBEDDING_MODEL_NAME, ALLOWED_EXTENSIONS, CHROMA_PERSIST_DIR

# --- Disable Telemetry/Warnings ---
os.environ["LLAMA_INDEX_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_doc_id(file_path: str) -> str:
    """Generate a consistent document ID from a file path."""
    return str(Path(file_path).resolve())

def build_persistent_index():
    """
    Reads files from the local directory and creates or updates a persistent
    ChromaDB vector index. It intelligently handles new, modified, and deleted files.
    """
    print("\n--- 1. Initializing Embeddings ---")
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        print("✅ Embeddings initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing HuggingFace Embeddings: {e}")
        sys.exit(1)

    print(f"\n--- 2. Loading or Creating Persistent Vector Index at '{CHROMA_PERSIST_DIR}' ---")
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("directory_analysis_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if the index is empty to decide between full build and update
    try:
        if chroma_collection.count() == 0:
            print("ℹ️ Index is empty. Performing a full build.")
            index_is_new = True
        else:
            print(f"✅ Existing index with {chroma_collection.count()} entries loaded successfully.")
            index_is_new = False
    except Exception as e:
        print(f"❌ Error checking index status: {e}")
        sys.exit(1)

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    print("\n--- 3. Scanning Directory and Updating Index ---")
    try:
        # Get file paths from the directory and add modification time to metadata
        # This function now correctly merges default metadata with custom metadata
        def file_metadata_func(file_path: str) -> dict:
            return {
                "file_path": file_path,
                "last_modified": os.path.getmtime(file_path)
            }

        fs_docs = SimpleDirectoryReader(
            input_dir=".", 
            required_exts=ALLOWED_EXTENSIONS, 
            recursive=True,
            file_metadata=file_metadata_func
        ).load_data()
        
        fs_doc_info = {get_doc_id(doc.metadata["file_path"]): doc for doc in fs_docs}

        if not fs_docs:
            print(f"⚠️ No files found in the current directory with extensions: {', '.join(ALLOWED_EXTENSIONS)}.")
            sys.exit(0)

        if index_is_new:
            # If the index is new, just add all documents
            print(f"Adding {len(fs_docs)} new documents to the index...")
            index.insert_nodes(Settings.node_parser.get_nodes_from_documents(fs_docs, show_progress=True))
        else:
            # Logic for updating an existing index
            # Get all document IDs currently in the index
            indexed_doc_ids = set(chroma_collection.get()['ids'])
            
            new_or_modified_docs = []
            for doc_id, doc in fs_doc_info.items():
                if doc_id not in indexed_doc_ids:
                    new_or_modified_docs.append(doc)
                else:
                    # Check modification time for existing docs
                    indexed_doc_mtime = vector_store.get_ref_doc_info([doc_id])[doc_id].get("last_modified")
                    if indexed_doc_mtime and doc.metadata["last_modified"] > indexed_doc_mtime:
                        # File has been modified, so we need to update it
                        index.delete_ref_doc(doc_id, delete_from_docstore=True)
                        new_or_modified_docs.append(doc)
            
            deleted_doc_ids = indexed_doc_ids - set(fs_doc_info.keys())

            # Process updates
            if new_or_modified_docs:
                print(f"Found {len(new_or_modified_docs)} new or modified documents to add/update.")
                index.insert_nodes(Settings.node_parser.get_nodes_from_documents(new_or_modified_docs, show_progress=True))
            else:
                print("No new or modified documents found.")

            if deleted_doc_ids:
                print(f"Found {len(deleted_doc_ids)} documents to delete.")
                for doc_id in deleted_doc_ids:
                    index.delete_ref_doc(doc_id, delete_from_docstore=True)
            else:
                print("No documents to delete.")

    except Exception as e:
        print(f"❌ Error during index update: {e}")
        sys.exit(1)

    print("\n✅ Index update process complete.")
    print("\n=======================================================")
    print("      Indexing is up-to-date. You can now run the analyzer.")
    print("=======================================================\n")

if __name__ == '__main__':
    build_persistent_index()