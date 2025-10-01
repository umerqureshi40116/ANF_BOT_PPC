# query_data.py
import streamlit as st
from dotenv import load_dotenv
import os
import re
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "anf-ppc-4"

# -------------------------------
# Connect to Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# -------------------------------
# Extract section number from generic queries
# -------------------------------
def extract_section_number(query: str):
    match = re.search(r"(?:section\s*)?(\d+[A-Z]?)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# -------------------------------
# Query Pinecone and merge results
# -------------------------------
def query_pinecone(query: str, k: int = 10):
    section_number = extract_section_number(query)
    merged_text = ""

    if section_number:
        results = vectorstore.similarity_search(
            query=section_number,
            k=k,
            filter={"section": section_number}
        )
        if results:
            # Merge all chunks of same section into single answer
            for res in results:
                text = getattr(res, "page_content", None)
                if not text:
                    text = res.metadata.get("text", "")
                merged_text += text.strip() + "\n\n"
            return merged_text.strip()
    
    # fallback to semantic search if no section or no results
    results = vectorstore.similarity_search_with_score(query, k=k)
    if results:
        for res, score in results:
            text = getattr(res, "page_content", None)
            if not text:
                text = res.metadata.get("text", "")
            merged_text += text.strip() + "\n\n"
        return merged_text.strip()

    return "⚠️ No results found."

# -------------------------------
# Streamlit UI with chat bubble CSS
# -------------------------------
st.set_page_config(page_title="ANF Educational Chatbot", layout="wide")

st.markdown("""
<style>
/* Chat CSS */
header, [data-testid="stSidebar"], footer, .stDeployButton, .stActionButton, .stStatusWidget, .stSpinner, .stNotification, .stToast {
    display: none !important;
}
[data-testid="stAppViewContainer"], .block-container {
    background: #f6fff8 !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
}
.chat { max-width: 600px; margin: 0 auto; padding-bottom: 80px; font-family: 'Segoe UI', 'Roboto', Arial, sans-serif; }
.header { width: 100%; background: #14532d; color: #fff; text-align: center; padding: 32px 0 20px 0; margin-bottom: 16px; border-radius: 0 0 24px 24px; box-shadow: 0 2px 8px rgba(20,83,45,0.08); }
.header img { width: 56px; height: 56px; border-radius: 14px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(20,83,45,0.10); }
.header .title h1 { font-size: 28px; margin: 0; font-weight: 700; letter-spacing: 1px; }
.message-row { display: flex; width: 100%; margin-bottom: 18px; }
.message-row.assistant { justify-content: flex-start; }
.message-row.user { justify-content: flex-end; }
.message-row.user .avatar { order: 2; margin-left: 5px; margin-right: 2px; }
.message-row.user .bubble.user { order: 1; }
.avatar { width: 36px; height: 36px; border-radius: 50%; background: #eee; overflow: hidden; box-shadow: 0 1px 4px rgba(20,83,45,0.10); flex-shrink: 0; margin-bottom: auto; margin-top: 2px; }
.avatar img { width: 100%; height: 100%; object-fit: cover; }
.bubble { background: #14532d; color: #fff; border-radius: 18px; padding: 16px 20px; max-width: 70%; font-size: 18px; line-height: 1.7; box-shadow: 0 2px 8px rgba(20,83,45,0.08); word-break: break-word; margin: 0 10px; text-align: left; box-sizing: border-box; overflow-wrap: anywhere; }
.bubble.user { background: #1e7c4c; }
.bubble.assistant { background: #14532d; }
.stChatInputContainer { background: #fff !important; border-top: 1px solid #e0e0e0 !important; box-shadow: 0 -2px 8px rgba(20,83,45,0.05); padding: 12px 0 !important; position: fixed; left: 0; right: 0; bottom: 0; z-index: 100; }
.stChatInput { background: #f6fff8 !important; border-radius: 18px !important; border: 2px solid #14532d !important; font-size: 18px !important; color: #14532d !important; padding: 14px 20px !important; margin: 0 auto !important; width: 95% !important; max-width: 570px !important; box-sizing: border-box; box-shadow: 0 2px 8px rgba(20,83,45,0.08); }
</style>
<div class="chat">
  <div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"/>
    <div class="title"><h1>ANF Academy Educational Chatbot</h1></div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Avatars
# -------------------------------
assistant_avatar_url = "https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"
user_avatar_url = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"

# -------------------------------
# Session state for chat
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
st.markdown('<div class="chat">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    avatar_url = user_avatar_url if role_class == "user" else assistant_avatar_url
    st.markdown(
        f'''
        <div class="message-row {role_class}">
            <div class="avatar"><img src="{avatar_url}" alt="{role_class} avatar"></div>
            <div class="bubble {role_class}">{msg["content"]}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)
# -------------------------------
# User input
# -------------------------------
if query := st.chat_input("Ask a legal question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Get assistant answer
    answer = query_pinecone(query)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display updated chat after input
st.markdown('<div class="chat">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    avatar_url = user_avatar_url if role_class == "user" else assistant_avatar_url
    st.markdown(
        f'''
        <div class="message-row {role_class}">
            <div class="avatar"><img src="{avatar_url}" alt="{role_class} avatar"></div>
            <div class="bubble {role_class}">{msg["content"]}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)
