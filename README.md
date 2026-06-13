# RAG Chatbot with LangChain

A simple Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, and HuggingFace embeddings.
The chatbot reads a PDF document and answers user questions based only on the content of that document.

---

## Features

* Load PDF documents
* Split text into chunks
* Generate embeddings using HuggingFace
* Store embeddings in FAISS vector database
* Perform semantic similarity search
* Answer questions from uploaded PDF
* "Not found in document" handling
* Beginner-friendly implementation

---

## Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Embeddings
* Sentence Transformers
* PyPDF
* VS Code

---

## Project Structure

```bash
rag-chatbot/
│
├── app.py
├── notes.pdf
├── README.md
├── requirements.txt
└── venv/
```

---

## Installation

### 1. Clone Repository

```bash
git clone <your-github-repo-link>
cd rag-chatbot
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
```

---

## Add PDF

Place your PDF file inside the project folder and rename it as:

```bash
notes.pdf
```

---

## Run Project

```bash
python app.py
```

---

## Sample Questions

* What is SQL?
* Explain DBMS
* What is primary key?
* Explain joins
* What is normalization?

---

## How It Works

1. Load PDF using PyPDFLoader
2. Split text into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Display answers from document

---

## RAG Workflow

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Vector Store
 ↓
Semantic Retrieval
 ↓
Answer Generation
```

---

## Future Improvements

* Streamlit UI
* Multiple PDF support
* Gemini/OpenAI integration
* Chat history
* Voice assistant
* Better response formatting

---

## Author

Dheeraj Kumar

* IIIT Bhagalpur
* Electronics and Communication Engineering
* MERN Stack & AI/ML Enthusiast
