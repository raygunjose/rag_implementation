import streamlit as st
import tempfile
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re

st.set_page_config(page_title="Smart RAG System", layout="wide")
st.title("Smart RAG (Clean Retrieval System)")

# -----------------------------
# Step 1: PDF Loader
# -----------------------------
# It converts a PDF file into plain text so your RAG system can process it.
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# -----------------------------
# Step 2: SMART chunking (FIXED)
# -----------------------------
def split_text(text):
    """
    Splits based on headings / policy sections
    """
    # Split when a new section starts (TITLE style lines)

    #     \n                → newline
    # (?=...)           → lookahead (don’t remove text, just detect)
    # [A-Z]             → starts with capital letter
    # [A-Za-z &]+       → words like "Car Rental Policy"
    # (\n|:)            → ends with newline OR colon
    chunks = re.split(r"\n(?=[A-Z][A-Za-z &]+(\n|:))", text)

    cleaned = []
    for c in chunks:
        c = c.strip()
        if len(c) > 80:   # avoid tiny junk chunks
            cleaned.append(c)

    return cleaned

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload Business Policy PDF", type="pdf")

if uploaded_file:
    # Creates a temporary file in your system. ../AppData/Local/Temp
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # This reads the entire uploaded file content into memory.
        tmp.write(uploaded_file.read())
        # saving temporary file object full path into file_path variable
        file_path = tmp.name

    text = load_pdf(file_path)
    chunks = split_text(text)

    st.success(f"Document loaded with {len(chunks)} clean sections")

    # -----------------------------
    # Step 3: TF-IDF Embeddings
    # RAG embedding step
    # TF-IDF vectorizer, converts text into numerical vectors 
        # so machines can compare meaning.
    # -----------------------------
    vectorizer = TfidfVectorizer(stop_words="english")
    # This does two things together: 
    # fit() - Learns vocabulary from your text chunks. 
    # transform() - Converts each chunk into a TF-IDF vector (numbers).
    X = vectorizer.fit_transform(chunks).toarray().astype("float32")

    # -----------------------------
    # Step 4: FAISS Index
    # -----------------------------
    # It creates vector database
    index = faiss.IndexFlatL2(X.shape[1])
    # stores all your document vectors into the FAISS index.
    index.add(X)

    # store in session
    # very important Streamlit concept — 
        # it’s what makes your RAG app “remember things”.
    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.index = index

    st.info("🚀 RAG System Ready")

# -----------------------------
# Step 5: Query Engine
# -----------------------------
query = st.text_input("Ask a question")

# checks two conditions before running the search:
# Did the user type a question?
# "index" in st.session_state
if query and "index" in st.session_state:
    # transform Uses the already trained vectorizer
    # toarray() - Converts sparse matrix → normal NumPy array
    # astype - Converts data type to 32-bit float
    # q_vec is a query vector
    q_vec = st.session_state.vectorizer.transform([query]).toarray().astype("float32")

    # It searches the FAISS index using your query vector
    # and returns the closest matching document chunks.
    # k=1 → best single match
    # k=3 → top 3 matches
    # k=5 → broader context
    # FAISS always returns two things:

    # D is Distances 
    # I is Index positions of matching items
    # I is the output from FAISS
    D, I = st.session_state.index.search(q_vec, k=1)
    
    # final retrieval step of RAG pipeline
    # Here we convert FAISS output back into real text.
    top_chunk = st.session_state.chunks[I[0][0]]

    # -----------------------------
    # DISPLAY
    # -----------------------------
    st.markdown("Relevant Policy Section")
    st.success(top_chunk)

    # -----------------------------
    # SIMPLE ANSWER GENERATION (RULE BASED)
    # -----------------------------
    st.markdown("Answer")

    answer = f"""
Based on the policy document:

{top_chunk}

This section directly answers your question about: "{query}"
"""
    st.write(answer)


# Requirements file:

# FAISS stands for Facebook AI Similarity Search.
# FAISS is a library for searching similar items in large datasets of vectors.

# what is faiss-cpu==1.8.0.post1
# 1-major version (big changes)
# 8-minor version (new features)
# 0-patch version (bug fixes)
# released after 1.8.0
# CPU-only version of FAISS

# scikit-learn
# A Python ML library used for text vectorization

# numpy<2.0
# many ML libraries are not compatible with NumPy 2.x yet.