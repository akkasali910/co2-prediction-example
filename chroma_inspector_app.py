import streamlit as st
import chromadb
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime

# --- Visualization Imports ---
import numpy as np
import plotly.express as px
import umap  # Requires: pip install umap-learn scikit-learn

# --- Configuration (from other project files) ---
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

COLLECTION_NAME = "directory_analysis_collection"

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="ChromaDB Inspector", layout="wide")

# --- Caching Functions ---

@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def get_chroma_collection():
    """Connects to the persistent ChromaDB and returns the collection object."""
    try:
        db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = db.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_PERSIST_DIR}'.")
        st.error(f"Error: {e}")
        st.warning("Please ensure you have run 'python build_index.py' first.")
        return None

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """Loads and caches the sentence-transformer model."""
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}'.")
        st.error(f"Error: {e}")
        return None

# --- Main Application ---

st.title("🔬 ChromaDB Vector Store Inspector")
st.markdown(f"Inspecting collection **`{COLLECTION_NAME}`** from path **`{CHROMA_PERSIST_DIR}`**")

collection = get_chroma_collection()
embed_model = get_embedding_model()

if collection is None or embed_model is None:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Collection Info")
    total_items = collection.count()
    st.metric("Total Items in Collection", total_items)
    st.info("This app allows you to browse the contents of the vector store and perform similarity searches.")

    st.divider()

    st.header("Export Collection")
    st.info("Export the entire collection to a file. This may be slow for very large collections.")

    # Use session state to store the prepared export data
    if "export_data" not in st.session_state:
        st.session_state.export_data = None
        st.session_state.export_format = None

    if st.button("Prepare Export File"):
        with st.spinner(f"Fetching all {total_items} items from ChromaDB..."):
            try:
                all_data = collection.get(include=["metadatas", "documents"])
                
                # Prepare data for both formats
                df = pd.DataFrame({
                    "id": all_data["ids"],
                    "document": all_data["documents"],
                    "metadata": [json.dumps(m) for m in all_data["metadatas"]]
                })
                st.session_state.export_data = {
                    "csv": df.to_csv(index=False).encode('utf-8'),
                    "json": df.to_json(orient='records', indent=2).encode('utf-8')
                }
                st.success("Export file is ready to download below.")
            except Exception as e:
                st.error(f"Failed to prepare export file: {e}")

    if st.session_state.export_data:
        st.download_button("Download as CSV", data=st.session_state.export_data["csv"], file_name="chroma_collection.csv", mime="text/csv")
        st.download_button("Download as JSON", data=st.session_state.export_data["json"], file_name="chroma_collection.json", mime="application/json")


# --- Tabs for different functionalities ---
tab_browse, tab_query, tab_visualize = st.tabs(["Browse All Items", "Query by Similarity", "Visualize Embeddings"])

with tab_browse:
    st.header("Browse Stored Items")
    
    if total_items == 0:
        st.warning("The collection is empty. No items to display.")
    else:
        # --- Deletion Section ---
        with st.expander("Delete an Item"):
            st.warning("⚠️ Deletion is permanent and cannot be undone.")
            delete_id = st.text_input("Enter the full ID of the item to delete:", placeholder="Copy an ID from the table below")
            
            if st.button("Delete Item by ID", type="primary", disabled=(not delete_id)):
                try:
                    # First, check if the item exists to provide a better error message
                    existing_item = collection.get(ids=[delete_id])
                    if not existing_item['ids']:
                        st.error(f"Item with ID '{delete_id}' not found in the collection.")
                    else:
                        collection.delete(ids=[delete_id])
                        st.success(f"Successfully deleted item: {delete_id}")
                        st.info("Refreshing page...")
                        st.rerun() # Rerun the script to update the view
                except Exception as e:
                    st.error(f"An error occurred during deletion: {e}")

        # --- Edit Section ---
        with st.expander("Edit an Item's Content"):
            st.info("ℹ️ Editing a document will also regenerate and update its vector embedding.")
            edit_id = st.text_input("Enter the ID of the item to edit:", key="edit_id_input", placeholder="Copy an ID from the table below")

            # Use session state to hold the item being edited. Fetch it when the ID changes.
            if edit_id and (st.session_state.get("item_to_edit_id") != edit_id):
                with st.spinner("Fetching item to edit..."):
                    item_data = collection.get(ids=[edit_id], include=["documents"])
                    if item_data['ids']:
                        st.session_state.item_to_edit_id = item_data['ids'][0]
                        st.session_state.item_to_edit_content = item_data['documents'][0]
                    else:
                        st.warning(f"Item with ID '{edit_id}' not found.")
                        # Clear session state if ID is not found
                        if "item_to_edit_id" in st.session_state:
                            del st.session_state.item_to_edit_id
                        if "item_to_edit_content" in st.session_state:
                            del st.session_state.item_to_edit_content
            
            # If an item is loaded into session state, show the editor
            if st.session_state.get("item_to_edit_id"):
                new_content = st.text_area(
                    "Document Content:", 
                    value=st.session_state.get("item_to_edit_content", ""), 
                    height=250,
                    key="edit_content_area"
                )

                if st.button("Save Changes", key="save_edit_button"):
                    with st.spinner("Updating document and regenerating embedding..."):
                        new_embedding = embed_model.encode(new_content).tolist()
                        collection.update(ids=[st.session_state.item_to_edit_id], documents=[new_content], embeddings=[new_embedding])
                        st.success(f"Successfully updated item: {st.session_state.item_to_edit_id}")
                        st.rerun()

        # --- Add New Item Section ---
        with st.expander("Add a New Item"):
            st.info("ℹ️ A vector embedding will be automatically generated for the new document.")
            with st.form(key="add_item_form", clear_on_submit=True):
                new_id = st.text_input("New Item ID:", placeholder="e.g., file_path/to/new_document.txt")
                new_content = st.text_area("Document Content:", height=150, placeholder="Enter the full text content of the new document.")
                new_metadata_str = st.text_input("Metadata (as JSON string):", placeholder='e.g., {"source": "manual_entry"}')
                
                submit_button = st.form_submit_button(label="Add New Item")

                if submit_button:
                    if not new_id or not new_content:
                        st.warning("Please provide both an ID and content for the new item.")
                    else:
                        try:
                            with st.spinner("Adding new item..."):
                                # Check if ID already exists
                                if collection.get(ids=[new_id])['ids']:
                                    st.error(f"Error: An item with ID '{new_id}' already exists.")
                                else:
                                    new_embedding = embed_model.encode(new_content).tolist()
                                    metadata = json.loads(new_metadata_str) if new_metadata_str else {}
                                    collection.add(ids=[new_id], documents=[new_content], embeddings=[new_embedding], metadatas=[metadata])
                                    st.success(f"Successfully added new item: {new_id}")
                                    st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format for metadata. Please check the syntax.")
                        except Exception as e:
                            st.error(f"An error occurred while adding the item: {e}")

        # Pagination controls
        st.subheader("All Stored Items")
        show_embeddings = st.checkbox("Show Vector Embeddings", value=False)
        
        page_size = st.number_input("Items per page:", min_value=1, max_value=100, value=10, key="browse_page_size")
        max_page = (total_items - 1) // page_size if total_items > 0 else 0
        page_number = st.number_input("Page:", min_value=0, max_value=max_page, value=0, key="browse_page_num")
        
        offset = page_number * page_size

        # Determine what to include in the get() call
        include_fields = ["metadatas", "documents"]
        if show_embeddings:
            include_fields.append("embeddings")

        # Fetch data from ChromaDB
        try:
            with st.spinner("Fetching data from ChromaDB..."):
                data = collection.get(
                    limit=page_size,
                    offset=offset,
                    include=include_fields
                )
            
            if not data['ids']:
                st.info("No data found for this page.")
            else:
                # Helper to format timestamp nicely
                def format_timestamp(ts):
                    if ts:
                        try:
                            return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                        except (TypeError, ValueError):
                            return "Invalid Timestamp"
                    return "N/A"

                # Create a DataFrame for better visualization
                df_data = {
                    "ID": data["ids"],
                    "Last Modified": [format_timestamp(m.get("last_modified")) for m in data["metadatas"]],
                    "Document": data["documents"],
                    "Metadata": [str(m) for m in data["metadatas"]]
                }
                # Explicitly check length to avoid numpy array truth value ambiguity
                if show_embeddings and "embeddings" in data and data["embeddings"] is not None and len(data["embeddings"]) > 0:
                    df_data["Embeddings (Vector)"] = [str(e[:5]) + '...' for e in data["embeddings"]] # Show a preview
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")

with tab_query:
    st.header("Query for Similar Documents")
    
    query_text = st.text_input("Enter your search query:", placeholder="e.g., what is the purpose of the python scripts?")
    n_results = st.slider("Number of results to retrieve:", min_value=1, max_value=20, value=5)

    if st.button("Search", disabled=(not query_text)):
        with st.spinner("Generating embedding and querying ChromaDB..."):
            try:
                # 1. Generate embedding for the query
                query_embedding = embed_model.encode(query_text).tolist()

                # 2. Query the collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"]
                )

                # 3. Display the results
                st.subheader("Query Results")
                if not results['ids'][0]:
                    st.info("No similar documents found.")
                else:
                    for i in range(len(results['ids'][0])):
                        with st.expander(f"**Result {i+1}** (Distance: {results['distances'][0][i]:.4f})"):
                            st.json(results['metadatas'][0][i])
                            st.markdown("---")
                            st.text_area("Document Content", value=results['documents'][0][i], height=200, disabled=True, key=f"query_doc_{i}")

            except Exception as e:
                st.error(f"An error occurred during the query: {e}")

with tab_visualize:
    st.header("Visualize Vector Embeddings in 2D")
    st.info("This tool uses UMAP to reduce the dimensionality of the embeddings to 2D for visualization.")
    st.warning("⚠️ This can be computationally intensive for very large collections.")

    max_items = collection.count()
    num_to_visualize = st.slider(
        "Number of items to visualize:", 
        min_value=min(50, max_items), 
        max_value=max_items, 
        value=min(1000, max_items),
        help="Select the number of the most recent items to process for visualization."
    )

    if st.button("Generate Visualization", key="generate_viz_button"):
        with st.spinner(f"Fetching {num_to_visualize} items and generating visualization..."):
            try:
                # 1. Fetch all data including embeddings
                data = collection.get(
                    limit=num_to_visualize,
                    include=["embeddings", "documents", "metadatas"]
                )

                # Explicitly check for existence and length to avoid numpy array truth value ambiguity
                embeddings = data.get("embeddings")
                if embeddings is None or len(embeddings) == 0:
                    st.warning("No embeddings found to visualize.")
                else:
                    # 2. Perform dimensionality reduction with UMAP
                    embeddings_array = np.array(data["embeddings"])
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    embedding_2d = reducer.fit_transform(embeddings_array)

                    # 3. Create a DataFrame for plotting
                    df_viz = pd.DataFrame(embedding_2d, columns=['x', 'y'])
                    df_viz['document'] = [doc[:100] + '...' if len(doc) > 100 else doc for doc in data['documents']]
                    df_viz['id'] = data['ids']

                    # 4. Create and display the interactive plot
                    fig = px.scatter(df_viz, x='x', y='y', hover_data=['id', 'document'], title="2D UMAP Projection of Document Embeddings")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during visualization: {e}")