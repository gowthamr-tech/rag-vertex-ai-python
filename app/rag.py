import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path

# Load environment variables from .env
load_dotenv()

def build_qa_chain():
    # Configuration from environment variables
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")

    # Setup Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    pdf_path = BASE_DIR / "data" / "sample.pdf"

    try:
        # 1. Load PDF
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # 2. Split Text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 3. Embeddings (Using Vertex AI mode)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            project=PROJECT_ID,
            location=LOCATION,
            vertexai=True
        )

        # 4. Create Vector DB
        vector_db = FAISS.from_documents(chunks, embeddings)

        # 5. Initialize LLM (Using Vertex AI mode)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.2,
            project=PROJECT_ID,
            location=LOCATION,
            vertexai=True
        )

        return vector_db, llm

    except Exception as e:
        print(f"Error during RAG chain build: {e}")
        raise e