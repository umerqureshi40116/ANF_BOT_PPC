################################ QUERY_DATA.PY ################################
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings  # updated class
from langchain_pinecone import PineconeVectorStore

# ------------------------------
# 1. Load API key & index name
# ------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "anf-ppc-4"

# ------------------------------
# 2. Connect to Pinecone
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# ------------------------------
# 3. Query Function
# ------------------------------
def query_pinecone(query: str, k: int = 5):
    """
    Perform semantic search over the entire PPC database.
    Returns top k results with metadata and text.
    """
    # If query is like 'section 171', do exact section filter
    import re
    section_match = re.search(r"section\s+(\d+[A-Z]?)", query, re.IGNORECASE)
    if section_match:
        section_number = section_match.group(1)
        print(f"🔎 Direct lookup for Section {section_number}")
        results = vectorstore.similarity_search(
            query=section_number,
            k=k,
            filter={"section": section_number}  # exact section
        )
        if results:
            return [(res, 1.0) for res in results]  # exact match
    # Otherwise do semantic search over entire content
    print(f"🔎 Semantic search for query: {query}")
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

# ------------------------------
# 4. Ask a question
# ------------------------------
if __name__ == "__main__":
    query = input("Enter your question about PPC: ")
    results = query_pinecone(query)

    if not results:
        print("⚠️ No results found.")
    else:
        print(f"\n🔍 Query Results:")
        print("="*80)
        for i, item in enumerate(results, 1):
            if isinstance(item, tuple):
                res, score = item
            else:
                res, score = item, 0.0

            text = res.metadata.get("text", "⚠️ No text available")
            source = res.metadata.get("source", "Unknown")
            section = res.metadata.get("section", "N/A")
            title = res.metadata.get("title", "N/A")

            print(f"\n📌 Result {i} | Score: {score:.3f}")
            print(f"Source: {source} | Section: {section} | Title: {title}")
            print("-"*80)
            print(text[:1500])  # preview first 1500 chars
            print("-"*80)
