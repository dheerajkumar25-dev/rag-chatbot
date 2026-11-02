
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.write("Generate academic abstracts and ask questions from your PDF documents.")

# Check API key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API key not found. Configure GROQ_API_KEY in your environment or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Cache embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# PDF upload
uploaded_file = st.file_uploader("Upload your research PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    try:
        with st.spinner("Processing your PDF..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                st.error("No readable content found in this PDF.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            docs = splitter.split_documents(documents)

            if not docs:
                st.error("Could not extract text from the uploaded PDF.")
                st.stop()

            vectorstore = FAISS.from_documents(docs, embeddings)

        st.success("PDF processed successfully!")

        # Generate abstract
        if st.button("Generate Abstract"):
            full_text = "\n".join(
                doc.page_content for doc in docs[:5]
            )

            abstract_prompt = f"""
Generate an academic-style abstract for the following document.

Document:
{full_text}

The abstract should be:
- Formal
- Concise
- Research-paper style
- Around 150-200 words
"""

            with st.spinner("Generating abstract..."):
                response = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": abstract_prompt}
                    ],
                    model=GROQ_MODEL,
                )

            st.subheader("Generated Abstract")
            st.write(response.choices[0].message.content)

        # Question answering
        st.subheader("Ask Questions About Your PDF")

        query = st.text_input("Enter your question")

        if query:
            with st.spinner("Searching your document..."):
                results = vectorstore.similarity_search(query, k=2)

                if not results:
                    st.warning("Not found in document")
                else:
                    context = "\n".join(
                        result.page_content for result in results
                    )

                    prompt = f"""
Answer the question only from the provided context.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say:
Not found in document
"""

                    response = client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=GROQ_MODEL,
                    )

                    st.subheader("Answer")
                    st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"An error occurred while processing your PDF: {e}")

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
