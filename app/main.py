import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tempfile
import shutil

from services.document_processor import DocumentProcessor
from services.vector_store import MilvusVectorStore
from services.llm_service import LangGraphLLMService
from services.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Document QA System")

# CORS para evitar problemas futuros
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serviços da API
vector_store = MilvusVectorStore()
document_processor = DocumentProcessor()
llm_service = LangGraphLLMService()
rag_pipeline = RAGPipeline(document_processor, vector_store, llm_service)


# Aqui, o único parâmetro obrigatório é o "question", o resto pode ser vazio
class QuestionRequest(BaseModel):
    session_id: str = "default"
    question: str
    chat_history: list = []


@app.post("/documents")
async def upload_documents(
    files: List[UploadFile] = File(...), session_id: str = "default"
):
    """
    Faz o upload de um ou mais PDFs.
    """

    if not files:
        raise HTTPException(status_code=400, detail="Nenhum PDF foi providenciado.")

    total_chunks = 0
    documents_indexed = 0

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        # Salva temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            chunks = await rag_pipeline.process_document(
                temp_file_path, file.filename, session_id
            )
            total_chunks += len(chunks)
            documents_indexed += 1
        finally:
            # Remove o arquivo temporário
            os.unlink(temp_file_path)

    if documents_indexed == 0:
        raise HTTPException(
            status_code=400, detail="Nenhum PDF válido foi providenciado"
        )

    return {
        "message": "Documentos processados com sucesso",
        "documents_indexed": documents_indexed,
        "total_chunks": total_chunks,
    }


@app.post("/question")
async def ask_question(request: QuestionRequest):
    """
    Envia uma mensagem para a API do RAG.
    """
    if not request.question:
        raise HTTPException(
            status_code=400, detail="O campo 'question' não pode estar vazio."
        )

    answer, context_chunks = await rag_pipeline.answer_question(
        request.question, request.chat_history, request.session_id
    )

    return {
        "answer": answer,
        "context_chunks": context_chunks,
    }
