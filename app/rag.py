from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path

def build_qa_chain():
    # 1. Setup Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    pdf_path = BASE_DIR / "data" / "sample.pdf"

    try:
        # 2. Load PDF
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # 3. Initialize Splitter (This was missing!)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # 4. Create Chunks
        chunks = splitter.split_documents(docs)

        # 5. Initialize Embeddings (Vertex AI mode)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            project="researchai-483206",
            location="us-central1",
            vertexai=True
        )

        # 6. Create Vector DB
        vector_db = FAISS.from_documents(chunks, embeddings)

        # 7. Initialize LLM (Vertex AI mode)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.2,
            project="researchai-483206",
            location="us-central1",
            vertexai=True
        )

        return vector_db, llm

    except Exception as e:
        print(f"Error during RAG chain build: {e}")
        raise e