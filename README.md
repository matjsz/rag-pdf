# Desafio de ML - RAG de PDFs

Esse é o repositório oficial para o desafio de ML para o RAG de PDFs. Siga a seção de instalação para rodar o projeto.

## Features

- Carregar documentos PDF e extrair o conteúdo
- Aplicar OCR em documentos digitalizados
- Chunkeniza texto de documentos para pesquisa semântica
- Armazena texto para recuperação posterior
- Swagger da API via FastAPI
- Frontend Streamlit para fácil interação

## Tech Stack

- **FastAPI**: Servidor com API REST para o backend do RAG
- **PyMilvus**: Banco de vetores para os embeddings
- **Streamlit**: Interface frontend
- **OpenAI API via LangGraph**: Orquestra a pipeline de RAG do LLM usado (OpenAI)
- **PyPDF2 & Tesseract OCR**: Extração de textos de PDFs
- **Docker**: Containerização

## Instalação

### Pré-requisitos

- Docker (opcional, mas extremamente recomendado)
- Python 3.12+
- Milvus database
- OpenAI API key

### Environment Variables

No arquivo `.env.prod` ajuste as seguintes variáveis:

```
OPENAI_API_KEY=chave-da-openai
OPENAI_MODEL_NAME=gpt-4o # ou outro modelo

MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530

API_URL=http://localhost:5000
```

### Opção 1 (recomendado): Docker

1. Instale o Milvus DB:

```bash
./install-milvus.sh
```

O container já vai ser iniciado logo em seguida.

1.1. Instale o Attu UI para visualização dos dados no Milvus DB (opcional):

```bash
./install-attu.sh
```

O container também já vai ser iniciado logo em seguida.

2. Instale o projeto:

```bash
./install.sh
```

Certifique-se de que o Milvus DB está rodando e inicie o container com:

```bash
./run.sh
```

### Opção 2: Local

1. Instale o Milvus DB:

```bash
./install-milvus.sh
```

O container já vai ser iniciado logo em seguida.

1.1. Instale o Attu UI para visualização dos dados no Milvus DB (opcional):

```bash
./install-attu.sh
```

2. Instale as dependências do projeto:

```bash
apt-get update

apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    python3-dev \
    gcc

pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

pip install poetry
```

```bash
poetry install
```

3. Inicie o servidor de API REST do RAG:

```bash
cd app

poetry run fastapi run main.py --port 5000
```

4. Inicie o servidor frontend do Streamlit:

```bash
poetry run streamlit run ui.py
```

## Detalhes de Uso

### API REST do RAG

- Você pode conferir todos os detalhes das rotas em: `http://localhost:5000/docs`

### Streamlit UI

- A interface frontend pode ser acessada em: `http://localhost:8501`

### Milvus DB

- O cluster do Milvus DB pode ser checado em: `http://localhost:9091/webui`
- Caso você tenha instalado o Attu UI, você pode checar todas as informações do Milvus em: `http://localhost:3000`, basta clicar em "Connect".

Agradeço pela oportunidade e pelo desafio! :)
