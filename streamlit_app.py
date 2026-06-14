
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import os

# LOAD ENV VARIABLES
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# GROQ CLIENT
client = Groq(
    api_key=groq_api_key
)

# PAGE TITLE
st.title("AI Research Assistant")
st.write("Generate Abstracts and Ask Questions from PDF")

# PDF UPLOAD
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:
    
    # SAVE PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # LOAD PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # SPLIT TEXT
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # CREATE EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # STORE IN FAISS
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    # ABSTRACT BUTTON
    if st.button("Generate Abstract"):
        full_text = ""
        for doc in docs[:5]:
            full_text += doc.page_content + "\n"
        abstract_prompt = f"""
        Generate an academic style abstract for the following document.
        Document:
        {full_text}
        The abstract should be:
        - formal
        - concise
        - research-paper style
        - around 150-200 words
        """
        abstract_response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": abstract_prompt,
                }
            ],

            model="llama-3.1-8b-instant",
        )
        st.subheader("Abstract")
        st.write(abstract_response.choices[0].message.content)

    # QUESTION ANSWERING
    query = st.text_input("Ask Question")
    if query:
        results = vectorstore.similarity_search(
            query,
            k=2
        )
        if len(results) == 0:
            st.write("Not found in document")
        else:
            context = ""
            for result in results:
                context += result.page_content + "\n"
            prompt = f"""
            Answer the question only from the provided context.
            Context:
            {context}
            Question:
            {query}
            If answer is not present in context,
            say:
            Not found in document
            """
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
            )
            st.subheader("Answer")
            st.write(chat_completion.choices[0].message.content)

