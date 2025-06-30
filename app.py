# 1. RAG-based MCQ Generator 
# Tech: LangChain + FAISS + AzureOpenAI + Streamlit 
# Goal: 
# • Upload a GenAI-related PDF 
# • Extract content, chunk & store in FAISS 
# • Use LangChain RetrievalQA + AzureOpenAI to generate: 
# o 5 MCQs with options and answers 
# o 5 Short Answer Questions 
# • Show output in Streamlit 

import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings  # Changed from OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Streamlit UI
st.set_page_config(page_title="RAG-Based MCQ Generator", layout="wide")
st.title("📘 RAG-Based MCQ Generator (GenAI PDFs)")

# Load environment variables
load_dotenv()

# File upload
uploaded_file = st.file_uploader("Upload a GenAI-related PDF file", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_faiss_index(chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("TEXTEMBEDDING_MODEL_NAME"),
        openai_api_version=os.getenv("TEXTEMBEDDING_API_VERSION"),
        azure_endpoint=os.getenv("TEXTEMBEDDING_API_BASE"),
        openai_api_key=os.getenv("TEXTEMBEDDING_API_KEY"),
        chunk_size=250,
    )
    return FAISS.from_texts(chunks, embeddings)

def get_qa_chain(retriever):
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model_name="gpt-4",
        temperature=0.7,    
    )

    prompt = PromptTemplate.from_template("""
You are an expert in Generative AI education. Based on the following context, generate:

1. Five Multiple Choice Questions (with 4 options each and correct answer).
2. Five Short Answer Questions.

Context:
{context}

Return the result clearly labeled as:
MCQs:
1. ...
Options: a) ... b) ... c) ... d) ...
Answer: ...

Short Answers:
1. ...
Answer: ...
""")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

# Workflow on PDF upload
if uploaded_file:
    with st.spinner("Extracting and processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        index = create_faiss_index(chunks)
        retriever = index.as_retriever()
        qa_chain = get_qa_chain(retriever)

    with st.spinner("Generating questions..."):
        output = qa_chain.run("Generate questions based on this content.")

    st.subheader("📋 Generated Output")
    st.text_area("Questions", value=output, height=600)
    
    st.success("Questions generated successfully!")