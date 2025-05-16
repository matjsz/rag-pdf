from typing import List, Dict, Any, Tuple
from .document_processor import DocumentProcessor
from .vector_store import MilvusVectorStore
from .llm_service import LangGraphLLMService
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_store: MilvusVectorStore,
        llm_service: LangGraphLLMService,
    ):
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def process_document(
        self, file_path: str, filename: str, session_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Processa um documento e armazena os embeddings e metadatas no Milvus DB

        Args:
            file_path (str): Caminho para o documento.
            filename (str): Nome do documento para metadata.
            session_id (str, opcional): Define o ID da coleção do Milvus, para poder começar uma conversa limpa na UI do Streamlit.
        """

        chunks = await self.document_processor.process_document(file_path, filename)

        await self.vector_store.insert_chunks(chunks, session_id)

        return chunks

    async def answer_question(
        self, question: str, chat_history: list, session_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Responde a mensagem do usuário usando a pipeline do RAG.

        Args:
            question (str): A mensagem do usuário.
            chat_history (list, opcional): Uma lista de mensagens como histórico de conversa para contexto conversacional.
            session_id (str, opcional): Define o ID da coleção do Milvus, para poder começar uma conversa limpa na UI do Streamlit.
        """

        question_embedding = self.document_processor.embeddings.embed_query(question)

        # Retrieval dos chunks mais relevantes
        context_chunks = await self.vector_store.search_similar_chunks(
            question_embedding, top_k=5, session_id=session_id
        )

        answer = await self.llm_service.generate_answer(
            question, chat_history, context_chunks
        )

        return answer, context_chunks
