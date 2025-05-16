import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    async def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extrai o texto de um documento PDF, também usando OCR se necessário.

        Args:
            file_path (str): Caminho para o documento PDF.
        """

        pdf = PdfReader(file_path)
        text = ""

        page_number = 1
        for page in pdf.pages:
            page_text = page.extract_text()

            # Faz o OCR caso não tenha texto
            if not page_text or len(page_text.strip()) < 50:
                # Converte a página para imagem e aplica o OCR
                images = convert_from_path(
                    file_path,
                    first_page=page_number,
                    last_page=page_number,
                )
                for img in images:
                    page_text = pytesseract.image_to_string(img)

            text += page_text + "\n\n"
            page_number += 1

        return text

    async def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Separa o texto em chunks.

        Args:
            text (str): Texto a ser chunkenizado.
            metadata (Dict[str, Any]): Metadata do chunk.
        """
        chunks = self.text_splitter.create_documents([text], [metadata])
        return [
            {"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
        ]

    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gera os embeddings para os chunks.

        Args:
            chunks (List[Dict[str, Any]]): Lista de chunks gerados com `DocumentProcessor.chunk_text`
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    async def process_document(
        self, file_path: str, filename: str
    ) -> List[Dict[str, Any]]:
        """
        Processa um documento PDF.

        Args:
            file_path (str): Caminho para o documento.
            filename (str): Nome do documento para metadata do chunk
        """
        text = await self.extract_text_from_pdf(file_path)
        metadata = {"source": filename}
        chunks = await self.chunk_text(text, metadata)
        embedded_chunks = await self.embed_chunks(chunks)
        return embedded_chunks
