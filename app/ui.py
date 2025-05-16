import streamlit as st
import requests
import os
import randomname
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Mestre dos PDFs", page_icon="📚", layout="centered")

# Endpoint da API do RAG
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Estilização, apenas alguns ajustes
st.markdown(
    """
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #262730;
        color: white;
    }
    .chat-message.assistant {
        background-color: #444656;
        color: white;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
    }
    .chat-message .message-content {
        margin-left: 1rem;
    }
    .stTextInput > div > div > input {
        padding: 0.75rem;
    }
    .css-1kyxreq {
        justify-content: center;
    }
    [data-testid=stMarkdownContainer] code {
        color: black;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Para o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_input" not in st.session_state:
    st.session_state.processed_input = ""

# O ID da sessão é gerado automaticamente ao carregar a página pela primeira vez
if "session_id" not in st.session_state:
    st.session_state.session_id = randomname.get_name(sep="_")


def display_chat_message(role, content):
    """
    Atualiza as mensagens da interface.

    Args:
        role (str): Para organização da interface, especificamente os icones.
        content (str): Conteúdo da mensagem.
    """
    if role == "user":
        with st.chat_message(
            "user",
            avatar="https://api.dicebear.com/9.x/glass/svg?seed=user&backgroundColor=d1d4f9&shape1[]",
        ):
            st.write("**Usuário**")
            st.markdown(content)
    else:
        with st.chat_message(
            "assistant",
            avatar="resources/icon.png",
        ):
            st.write("**Mestre dos PDFs**")
            st.markdown(content)


def state_messages_to_list():
    """
    Retorna um histórico de chat formatado para melhor uso da API do RAG.
    """

    formated_messages = []
    for history in st.session_state.messages:
        formated_messages.append(history["content"])

    # Limita até 5 mensagens no histórico (5 do usuário e 5 da IA)
    if len(formated_messages) > 10:
        formated_messages = formated_messages[-10:]

    return formated_messages


def ask_question(question):
    """
    Faz uma pergunta à API, ou seja, envia uma mensagem.

    Args:
        question (str): Mensagem a ser enviada.
    """
    try:
        # Adiciona no histórico de mensagens
        st.session_state.messages.append({"role": "user", "content": question})

        response = requests.post(
            f"{API_URL}/question",
            json={
                "session_id": st.session_state.session_id,
                "question": question,
                "chat_history": state_messages_to_list(),
            },
        )
        response.raise_for_status()
        response_text = response.json()["answer"]

        # Adiciona no histórico de mensagens
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao gerar resposta: {str(e)}")
        return None


def handle_user_input():
    """
    Processa a mensagem do usuário e pega a resposta da API do RAG.
    """
    question = st.session_state.user_input

    with st.spinner("Gerando resposta..."):
        result = ask_question(question)

        if result:
            # Armazena o contexto dos chunks para poder ver na UI do Streamlit
            st.session_state.last_context = result.get("context_chunks", [])


def upload_documents(files):
    """
    Faz o upload dos PDFs.

    Args:
        files: PDFs a serem enviados para o Milvus DB.
    """
    if not files:
        return None

    files_data = [("files", file) for file in files]

    try:
        response = requests.post(f"{API_URL}/documents", files=files_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao processar documento: {str(e)}")
        return None


# UI:

st.markdown(
    """
    <h1 style="text-align: center">O Mestre dos PDFs 🧙</h1>
    <p style="text-align: center">Faça perguntas sobre seus PDFs!</p>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """,
    unsafe_allow_html=True,
)


# Dialog do contexto dos chunks
@st.dialog("Contexto do Mestre dos PDFs")
def context_dialog():
    last_message = st.session_state.messages[-2]["content"]

    st.header("Contexto usado para:")
    st.markdown(f"`{last_message}`")

    if hasattr(st.session_state, "last_context") and st.session_state.last_context:
        for i, chunk in enumerate(st.session_state.last_context):
            with st.expander(
                f"Chunk {i + 1} do documento {chunk['source']} (Score: {chunk['score']:.4f})"
            ):
                st.markdown(chunk["text"])
    else:
        st.info("Faça o upload de PDFs e/ou faça uma pergunta para ver o contexto.")


# Sidebar
with st.sidebar:
    st.text_input(key="session_id", label="ID da Sessão")

    st.header("Upload de PDFs")
    if len(st.session_state.messages) > 1:
        if st.button("Ver contexto", type="secondary"):
            context_dialog()
    uploaded_files = st.file_uploader(
        "Deixe seus PDFs aqui e veja a mágica acontecer!",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if uploaded_files:
        if st.button("Processar PDFs"):
            with st.spinner("Processando PDFs..."):
                result = upload_documents(uploaded_files)
                if result:
                    st.success(
                        f"{result['documents_indexed']} documentos processados com sucesso ({result['total_chunks']} chunks)"
                    )

# Mantém o chat atualizado
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

st.chat_input(
    key="user_input",
    placeholder="Me pergunte algo...",
    on_submit=handle_user_input,
)
