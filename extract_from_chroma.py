#!/usr/bin/env python3
"""
This script demonstrates how to extract documents, metadata, and vector embeddings
from an in-memory ChromaDB collection.
"""
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Creates a sample in-memory ChromaDB index, adds data, and then extracts it.
    """
    # --- 1. Setup a Sample In-Memory ChromaDB ---
    logging.info("Setting up in-memory ChromaDB client and collection...")
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_in_memory_collection")

    # --- 2. Add Sample Data (Simulating your app's indexing) ---
    logging.info("Adding sample documents to the collection...")
    documents = [
        "The train.py script is used for model training.",
        "The evaluate.py script calculates model performance metrics like R2 and MSE.",
        "The preprocess.py script handles feature engineering."
    ]
    metadatas = [
        {"source_file": "train.py"},
        {"source_file": "evaluate.py"},
        {"source_file": "preprocess.py"}
    ]
    ids = ["doc1", "doc2", "doc3"]

    # Generate embeddings for the documents
    logging.info("Generating embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(documents).tolist()

    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    logging.info(f"Successfully added {collection.count()} documents to the index.")

    # --- 3. Extract Data from the ChromaDB Collection ---
    logging.info("\n--- Starting Extraction Process ---")
    
    # The key step: use collection.get() to retrieve all data.
    # We specify `include` to ensure we get the embeddings, documents, and metadata.
    extracted_data = collection.get(
        include=["embeddings", "documents", "metadatas"]
    )

    logging.info(f"Successfully extracted {len(extracted_data['ids'])} items from the collection.")

    # --- 4. Process and Display the Extracted Data ---
    # The extracted data can now be formatted and sent to another vector database.
    output_for_other_db = []
    for i in range(len(extracted_data["ids"])):
        record = {
            "id": extracted_data["ids"][i],
            "text_content": extracted_data["documents"][i],
            "metadata": extracted_data["metadatas"][i],
            "vector": extracted_data["embeddings"][i] # The crucial vector embedding
        }
        output_for_other_db.append(record)

    logging.info("\n--- Extracted Data (Ready for another DB) ---")
    print(json.dumps(output_for_other_db, indent=2))

if __name__ == "__main__":
    main()