# diagnostic.py
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "anf-ppc-4"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Check what sections exist in the database
print("Checking database contents...")
stats = index.describe_index_stats()
print(f"Index stats: {stats}")

# Try to fetch vectors with section 171
print("\nSearching for section 171...")
query_response = index.query(
    vector=[0.1] * 768,  # dummy vector
    top_k=10,
    include_metadata=True,
    filter={"section": "171"}
)

if query_response.matches:
    print(f"Found {len(query_response.matches)} matches for section 171:")
    for match in query_response.matches:
        print(f"ID: {match.id}")
        print(f"Section: {match.metadata.get('section', 'N/A')}")
        print(f"Title: {match.metadata.get('title', 'N/A')}")
        print(f"Text preview: {match.metadata.get('text', 'N/A')[:200]}...")
        print("---")
else:
    print("No matches found for section 171")
    
    # Let's see what sections ARE in the database
    print("\nSearching for any sections to see what exists...")
    all_response = index.query(
        vector=[0.1] * 768,
        top_k=5,
        include_metadata=True
    )
    print("Some sections in database:")
    for match in all_response.matches:
        print(f"Section: {match.metadata.get('section', 'N/A')} - {match.metadata.get('title', 'N/A')[:50]}...")