import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma



# 1. Load .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 2. Load PDF
pdf_path = "https://new.kenyalaw.org/akn/ke/act/2010/constitution/eng@2010-09-03/source"
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document chunks from the constitution PDF.")

# 3. Parent-Child Split
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
parents = parent_splitter.split_documents(docs)

# 4. Embedding Model
bge_model = SentenceTransformer("BAAI/bge-base-en")

class BGEEmbedding:
    def embed_documents(self, texts):
        return bge_model.encode(texts, batch_size=8, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return bge_model.encode([text], normalize_embeddings=True).tolist()[0]

# 5. Vectorstore + Docstore
vectorstore = Chroma(
    collection_name=f"kenya_constitution_{str(uuid.uuid4())}",
    embedding_function=BGEEmbedding(),
    persist_directory="./kenya_constitution_chroma"
)

docstore = InMemoryStore()

# 6. ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter
)

# 7. Ingest to vector DB
print("Ingesting documents...")
retriever.add_documents(docs)

# 8. Chat model
chat_model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

# 9. Prompt Template
template = """[INST] <<SYS>> Use the following context to answer the question. 
Be precise. Answer the questions based on the given context only. Do not use your knowledge. If a questions lies outside the ontext given, say "I am a chatbot that helps humans understand the Kenyan Constitution. Kindly ask me a question about the Kenyan Constitution."  <</SYS>>
{context}

Question: {question}
Answer:[/INST]
"""
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# 10. RAG Chain (Parent-aware)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True
)

# 11. FastAPI Setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(payload: Question):
    result = qa_chain(payload.query)
    return {
        "question": payload.query,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }

