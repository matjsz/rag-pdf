import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List, Dict, Any
import uuid
from dotenv import load_dotenv

load_dotenv()


class MilvusVectorStore:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        connections.connect(
            "default",
            host=self.host,
            port=self.port,
        )
        self.dim = 1536  # A OpenAI faz o uso de vetores de embedding com dimensão 1536
        self.collection = None

    def _create_collection_if_not_exists(self, session_id: str = "default"):
        """
        Cria a coleção do Milvus DB caso não exista.

        Args:
            session_id (str, opcional): Define o ID da coleção do Milvus, para poder começar uma conversa limpa na UI do Streamlit.
        """

        if not self.collection:
            if utility.has_collection(session_id):
                self.collection = Collection(session_id)
            else:
                # Schema da coleção
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=36,
                    ),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(
                        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim
                    ),
                ]
                schema = CollectionSchema(fields=fields)

                # Instancia e cria a coleção
                self.collection = Collection(name=session_id, schema=schema)

                # Cria o index para buscas
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64},
                }
                self.collection.create_index(
                    field_name="embedding", index_params=index_params
                )

    async def insert_chunks(
        self, chunks: List[Dict[str, Any]], session_id: str = "default"
    ) -> int:
        """
        Insere chunks de um documento no Milvus DB.

        Args:
            chunks (List[Dict[str, Any]]): Chunks do documento PDF.
            session_id (str, opcional): Define o ID da coleção do Milvus, para poder começar uma conversa limpa na UI do Streamlit.
        """

        self._create_collection_if_not_exists(session_id)

        # Prepara os dados no schema definido
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        texts = [chunk["text"] for chunk in chunks]
        sources = [chunk["metadata"]["source"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]

        # Inserção
        entities = [ids, texts, sources, embeddings]
        self.collection.insert(entities)

        self.collection.load()

        return len(chunks)

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        session_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        Busca por chunks mais relevantes de acordo com o embedding da mensagem do usuário.

        Args:
            query_embedding (List[float]): Vetor com o embedding da mensagem do usuário.
            top_k (int): Define quantos chunks serão retornados para contexto do RAG.
            session_id (str, opcional): Define o ID da coleção do Milvus, para poder começar uma conversa limpa na UI do Streamlit.
        """

        self._create_collection_if_not_exists(session_id)
        self.collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"],
        )

        # Formata os resultados para melhor uso futuro
        chunks = []
        for hits in results:
            for hit in hits:
                chunks.append(
                    {
                        "text": hit.entity.get("text"),
                        "source": hit.entity.get("source"),
                        "score": hit.score,
                    }
                )

        return chunks
