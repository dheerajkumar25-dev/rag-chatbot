
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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


# ---------------- PRINT ANSWER ----------------

if len(results) == 0:
    print("Not found in document")

else:
    print("\nANSWER:\n")
    for result in results:
        print(result.page_content)

