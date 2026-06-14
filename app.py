
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
# LOAD PDF
loader = PyPDFLoader("notes.pdf")
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
# MENU
print("\n===== AI RESEARCH ASSISTANT =====")
print("1. Generate Abstract")
print("2. Ask Question")
choice = input("\nEnter Choice : ")

# ABSTRACT GENERATION
if choice == "1":
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
    print("\nABSTRACT:\n")
    print(abstract_response.choices[0].message.content)

# QUESTION ANSWERING
elif choice == "2":
    query = input("\nAsk Question : ")
    # ---------------- SEARCH RELEVANT CHUNKS ----------------

    results = vectorstore.similarity_search(
        query,
        k=2
    )
    #HANDLE NO RESULTS 
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
        #GROQ RESPONSE
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
# INVALID INPUT
else:
    print("Invalid Choice")

