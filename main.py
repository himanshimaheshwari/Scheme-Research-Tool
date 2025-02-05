import openai
import streamlit as st
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# download and read PDF content from a URL
def download_pdf_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("temp_downloaded.pdf", "wb") as f:
                f.write(response.content)
            reader = PdfReader("temp_downloaded.pdf")
            pdf_text = "".join([page.extract_text() for page in reader.pages])
            st.write(f"Fetched content from {url[:50]}...")  # Display a preview of the URL
            return pdf_text
        else:
            st.error("Failed to download PDF. Status code: {}".format(response.status_code))
            return None
    except Exception as e:
        st.error(f"An error occurred while downloading the PDF: {str(e)}")
        return None

# generate embeddings for a given text
def generate_embeddings(text):
    try:
        return model.encode(text)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# create and store a FAISS index
def save_faiss_index(embeddings, docs, file_name="faiss_index.pkl"):
    vector_dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(vector_dim)
    index.add(np.array(embeddings).astype("float32"))
    with open(file_name, "wb") as f:
        pickle.dump((index, docs), f)
    st.write(f"FAISS index saved as '{file_name}'")

#  load an existing FAISS index
def load_faiss_index(file_name="faiss_index.pkl"):
    with open(file_name, "rb") as f:
        index, docs = pickle.load(f)
    return index, docs

# query the FAISS index
def query_index(index, query_embedding, docs, top_k=3):
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    results = [(docs[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# Streamlit interface for URL input and querying
st.title("Government Scheme Research Tool")
st.sidebar.subheader("Provide URL(s) for PDF files")

input_urls = st.sidebar.text_area("Enter one URL per line:")

if st.sidebar.button("Process URLs"):
    urls = input_urls.splitlines()
    docs, embeddings = [], []

    for url in urls:
        content = download_pdf_content(url)
        if content:
            docs.append(content)
            embed = generate_embeddings(content)
            if embed is not None:
                embeddings.append(embed)
    
    if embeddings:
        save_faiss_index(embeddings, docs)
        st.success("PDFs processed and indexed successfully!")

# Question input and FAISS index query
question = st.text_input("Enter your question about the schemes:")
if question and st.button("Submit Query"):
    query_embedding = generate_embeddings(question)
    
    try:
        index, docs = load_faiss_index()
        results = query_index(index, query_embedding, docs)

        st.write("Top Relevant Results:")
        for i, (doc, score) in enumerate(results):
            st.write(f"Result {i+1} (Similarity Score: {score:.2f}):")
            st.write(doc[:500])  # Display a snippet of the document
            st.write("-" * 50)

    except FileNotFoundError:
        st.error("FAISS index file not found. Please process URLs first.")
