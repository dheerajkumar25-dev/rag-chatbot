# AI Research Assistant using RAG

An AI-powered Research Assistant built using Retrieval-Augmented Generation (RAG), FAISS Vector Database, HuggingFace Embeddings, Groq LLM, and Streamlit.

The application allows users to upload PDF documents, generate academic-style abstracts, and ask intelligent questions based on document content using semantic search.

---

# Features

* Upload any PDF document
* Generate academic-style abstracts
* Ask questions from uploaded PDF
* Semantic similarity search using FAISS
* HuggingFace embeddings for vector generation
* Groq LLM integration for intelligent responses
* Streamlit web interface
* Secure API key handling using `.env`
* Dynamic PDF processing

---

# Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Sentence Transformers
* Groq API
* PyPDF
* dotenv

---

# Project Structure

```bash
rag-chatbot/
│
├── app.py
├── streamlit_app.py
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── temp.pdf
└── venv/
```

---

# Project Workflow

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings Generation
↓
FAISS Vector Database
↓
Semantic Retrieval
↓
Groq LLM
↓
Abstract Generation / Question Answering

---

# Installation

## Clone Repository

```bash
git clone https://github.com/dheerajkumar25-dev/rag-chatbot.git
```

---

## Navigate to Project Folder

```bash
cd rag-chatbot
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Setup Environment Variables

Create a `.env` file and add:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Run Streamlit Application

```bash
streamlit run streamlit_app.py
```

---

# Sample Questions

* What is SQL?
* Explain DBMS
* What is normalization?
* Explain joins
* What are primary keys?

---

# Core Concepts Used

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Embeddings
* Large Language Models (LLMs)
* Prompt Engineering

---

# Future Improvements

* Multiple PDF support
* Chat history
* Voice interaction
* Cloud deployment
* Dark mode UI
* PDF highlighting

---

# Author

Dheeraj Kumar

* IIIT Bhagalpur
* Electronics and Communication Engineering
* MERN Stack & AI/ML Enthusiast

GitHub:
https://github.com/dheerajkumar25-dev
