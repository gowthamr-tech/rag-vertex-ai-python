import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

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

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

        # 1. Initialize Pinecone Client
        pc = Pinecone(api_key=PINECONE_API_KEY)

       # 2. Check/Create Index
        if INDEX_NAME in pc.list_indexes().names():
            # If the dimension is wrong (1024), we must delete it
            desc = pc.describe_index(INDEX_NAME)
            if desc.dimension != 3072: # Change this to 3072
                print(f"Deleting index {INDEX_NAME} due to dimension mismatch...")
                pc.delete_index(INDEX_NAME)

        # Create a fresh index with the correct dimensions
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=3072, # Match Gemini's 3072 output
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        # 1. Load PDF
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # 1. Setup Postgres Connection
        # Format: postgresql+psycopg2://user:password@host:port/dbname
        db_uri = "postgresql+psycopg2://postgres:1234@localhost:5432/testing_data"
        db=SQLDatabase.from_uri(db_uri)
        # 2. Corrected SQL Loader
        sql_loader = SQLDatabaseLoader(
            # We use a query to combine columns into a single descriptive string
         query="SELECT id, 'Person: ' || person_name || ' booked the movie: ' || movie_name as booking_info FROM public.ticket_bookings",
         db=db,
        #  page_content_col="booking_info" # This matches the alias in our query above
        )     
        sql_docs=sql_loader.load()
        

        web_links=["https://in.bookmyshow.com/explore/home/chennai","https://tickets.chennaimetrorail.org/onlineticket","https://chennaimetrorail.org/phonepe/","https://www.zomato.com/chennai/restaurants/on/east-coast-road-ecr?page=6&order-online=1"]
        web_loader=WebBaseLoader(web_links)
        web_docs=web_loader.load()
        all_docs = web_docs+docs+sql_docs
        # +sql_docs

        # 2. Split Text
    
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
       
        # 3. Embeddings (Using Vertex AI mode)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            project=PROJECT_ID,
            location=LOCATION,
            vertexai=True
        )
        # # 4. Create Vector DB
        # vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=INDEX_NAME
            )
        # # Get Actual vector index
        # index=vector_db.index
        # # vector_db.save_local("my_index")
        # first_vector=index.reconstruct(0)
        # print("First vector (embedding) for the first chunk:", first_vector[:10])
        # 5. Initialize LLM (Using Vertex AI mode)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.2,
            project=PROJECT_ID,
            location=LOCATION,
            vertexai=True
        )
        # print("Inspect the vector store")
        # num_chunks=len(vector_db.index_to_docstore_id)
        # for i in range(min(5, num_chunks)):
        #     docid = vector_db.index_to_docstore_id[i]
        #     document=vector_db.docstore.search(docid)
        #     print(f"Document {i+1}: {document.page_content[:200]}...")  # Print first 200 characters of the document
        return vector_db, llm

    except Exception as e:
        print(f"Error during RAG chain build: {e}")
        raise e