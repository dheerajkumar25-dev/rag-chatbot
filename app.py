
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---------------- GROQ API ----------------

client = Groq(
    api_key="YOUR_GROQ_API_KEY"
)

# ---------------- LOAD PDF ----------------

loader = PyPDFLoader("notes.pdf")
documents = loader.load()

# ---------------- SPLIT TEXT ----------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# ---------------- CREATE EMBEDDINGS ----------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- STORE IN FAISS ----------------

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# ---------------- USER QUESTION ----------------

query = input("Ask Question : ")

# ---------------- SEARCH RELEVANT CHUNKS ----------------
results = vectorstore.similarity_search(
    query,
    k=2
)


# ---------------- HANDLE NO RESULTS ----------------

if len(results) == 0:
    print("Not found in document")

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
    # ---------------- GROQ RESPONSE ----------------

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-8b-instant",
    )
    print("\nANSWER:\n")
    print(chat_completion.choices[0].message.content)

