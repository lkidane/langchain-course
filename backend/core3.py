from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

# -----------------------------
# ✅ EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1"
)

# -----------------------------
# ✅ VECTOR STORE
# -----------------------------
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings
)

# -----------------------------
# ✅ MODEL (OLLAMA)
# -----------------------------
model = ChatOllama(
    model="mymodel",   # change to llama3 or mistral if needed
    temperature=0
)

# -----------------------------
# ✅ TOOL: RETRIEVAL
# -----------------------------
@tool
def retrieve_context(query: str) -> str:
    """
    
    REQUIRED tool. MUST be used before answering.
    Retrieves relevant LangChain documentation.

    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

    return serialized

# -----------------------------
# ✅ MAIN FUNCTION
# -----------------------------
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline using initialize_agent.
    """

    # system_prompt = (
    #     "You are a LangChain documentation assistant.\n\n"
    #     "You MUST ALWAYS call the 'retrieve_context' tool before answering.\n"
    #     "Do NOT answer from your own knowledge.\n\n"
    #     "Follow this EXACT format:\n\n"
    #     "Thought: ...\n"
    #     "Action: retrieve_context\n"
    #     "Action Input: <user query>\n"
    #     "Observation: <tool result>\n"
    #     "Final Answer: <answer based ONLY on retrieved docs>\n\n"
    #     "If you skip the Action step, your answer is invalid."
    # )
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = initialize_agent(
        tools=[retrieve_context],
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": system_prompt
        },
    )

    # ✅ Run agent
    response = agent.invoke(query)

    # ✅ Extract answer
    answer = response.get("output", "")

    return {
        "answer": answer
    }

# -----------------------------
# ✅ ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    result = run_llm("what are deep agents?")

    print("\n✅ ANSWER:\n")
    print(result["answer"])