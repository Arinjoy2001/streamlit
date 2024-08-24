# Streamlit app.py

import streamlit as st
import docx2txt
from pdfminer.high_level import extract_text
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Set OpenAI API key - Replace with your actual key!
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" 

def parse_and_preprocess(file, file_extension):
    """Parses and preprocesses the uploaded file based on its extension."""
    text = ""
    if file_extension == "txt":
        text = file.read().decode("utf-8")  
    elif file_extension == "pdf":
        text = extract_text(file)  
    elif file_extension == "docx":
        text = docx2txt.process(file)  
    return text

def create_knowledge_base(text):
    """Creates a Chroma knowledge base from the given text."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(text)
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    return retriever

def generate_response(query, retriever):
    """Generates a response using the RetrievalQA chain."""
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa.run(query)

# Streamlit app interface
st.title("Document Q&A Chat")

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = file_name.split(".")[-1]
        processed_text = parse_and_preprocess(uploaded_file, file_extension)
        all_text += processed_text + "\n"
    retriever = create_knowledge_base(all_text)

if "retriever" in locals():
    user_query = st.text_input("Ask a question:")
    if user_query:
        response = generate_response(user_query, retriever)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write("**User:**", message["content"])
        else:
            st.write("**Assistant:**", message["content"])
