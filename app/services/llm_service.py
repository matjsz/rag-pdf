import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class RAGState(TypedDict):
    question: str
    chat_history: list
    context: List[Dict[str, Any]]
    answer: str


class LangGraphLLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Constrói um graph do LangGraph para a pipeline de RAG.
        """

        # Define o esquema do graph
        workflow = StateGraph(state_schema=RAGState)

        def generate_answer(state):
            """
            Gera uma resposta com base no contexto e a mensagem.

            Args:
                state: State no formato de `RAGState` para geração da resposta.
            """
            context = state["context"]
            chat_history = state["chat_history"]
            question = state["question"]

            context_text = "\n\n".join(
                [
                    f"Fonte para o texto abaixo: {chunk['source']}\n---\n{chunk['text']}"
                    for chunk in context
                ]
            )
            chat_history_text = "\n".join(chat_history)
            system_message = SystemMessage(
                content=f"""Você é um assistente útil chamado 'Mestre dos PDFs' que responde às perguntas com base no contexto.
            
Chat History:
{chat_history_text}

Context:
{context_text}

Responda à mensagem, caso seja uma pergunta, com base SOMENTE no contexto fornecido. Se o contexto ou o histórico do chat não contiver a resposta, diga "Ops! Não tenho informações suficientes para fazer minha mágica dos PDFs :(". Caso contrário, se não for uma pergunta, responda de forma conversacional e gentil.
Seja conciso e direto.
Ao fim da mensagem, inclua uma seção de 'Referências', citando as fontes que você usou para responder, se possível.
"""
            )

            human_message = HumanMessage(content=question)

            response = self.llm.invoke([system_message, human_message])

            return {"answer": response.content}

        # Workflow simples apenas com pergunta -> geração de resposta -> fim
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_edge("generate_answer", END)

        workflow.set_entry_point("generate_answer")

        return workflow.compile()

    async def generate_answer(
        self, question: str, chat_history: list, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Gera uma resposta com base no contexto e a mensagem.

        Args:
            question (str): A mensagem do usuário.
            chat_history (list, opcional): Uma lista de mensagens como histórico de conversa para contexto conversacional.
            context_chunks (List[Dict[str, Any]]): Contexto para RAG do(s) documento(s) PDF.
        """

        input_state = {
            "question": question,
            "chat_history": chat_history,
            "context": context_chunks,
            "answer": "",
        }

        result = self.graph.invoke(input_state)

        return result["answer"]
