# 📘 RAG-Based MCQ Generator

🚀 **Live Demo**: [Click here to try the app](https://mcq-generator-ehcuhgb7gpbfg7g2.centralindia-01.azurewebsites.net/)

A Streamlit-based web app that leverages Retrieval-Augmented Generation (RAG) using **LangChain**, **FAISS**, and **Azure OpenAI** to automatically generate questions from Generative AI-related PDFs.

This tool is designed for educators, learners, and AI enthusiasts who want to quickly extract knowledge and convert it into Multiple Choice Questions (MCQs) and Short Answer Questions.

---

## 🎯 Features

✅ Upload a **GenAI-related PDF**  
✅ Extract content and intelligently chunk the text  
✅ Embed and index using **FAISS** and **Azure OpenAI Embeddings**  
✅ Use **LangChain’s RetrievalQA** to query with context  
✅ Auto-generate:
- 5 **Multiple Choice Questions** (MCQs) with 4 options each
- 5 **Short Answer Questions**

✅ Streamlit-powered interface to preview questions

---

## 🧠 Technologies Used

| Component      | Technology        |
|----------------|-------------------|
| Language Model | Azure OpenAI (GPT-4) |
| Embedding Model| Azure OpenAI Embeddings |
| Vector DB      | FAISS             |
| Text Splitter  | LangChain RecursiveCharacterTextSplitter |
| Framework      | Streamlit         |
| PDF Parsing    | PyPDF2            |
| Env Management | python-dotenv     |

---

## 📁 Folder Structure
```
rag-mcq-generator/
│
├── app.py # Main Streamlit application
├── .env # API keys and configurations
├── requirements.txt # All dependencies
├── README.md # This file
└── screenshots/ # UI previews (optional)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-mcq-generator.git
cd rag-mcq-generator
```
```
streamlit run app.py
```
