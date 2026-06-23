import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain import hub
# from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


#initialize the inxex as same
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1"
)

# initialize the vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index" , embedding=embeddings
)

# model = init_chat_model("mymodel", model_provider="langchain-ollama")
#Ollama model
model = ChatOllama(
    model="mymodel",   # your custom Ollama model name
    temperature=0
)

# @tool(response_format="content_and_artifact")
# def retrieve_context(query: str):
#     """Retrieve relevant documentation to help answer user queries about LangChain."""
#     # Retrieve top 4 most similar documents
#     retrieved_docs = vectorstore.as_retriever().invoke(query, k=2)
    
#     # Serialize documents for the model
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
    
#     # Return both serialized content and raw documents
#     return serialized, retrieved_docs


# ✅ GLOBAL variable
LAST_RETRIEVED_DOCS = []

# @tool(response_format="content_and_artifact")
# def retrieve_context(query: str):
#     """Retrieve relevant documentation for answering questions."""
    
#     global LAST_RETRIEVED_DOCS

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
#     retrieved_docs = retriever.invoke(query)

#     # ✅ store globally
#     LAST_RETRIEVED_DOCS = retrieved_docs

#     MAX_CHARS = 1000

#     serialized = "\n\n".join(
#         f"Source: {doc.metadata.get('source', 'Unknown')}\n"
#         f"Content: {doc.page_content[:MAX_CHARS]}"
#         for doc in retrieved_docs
#     )

#     return serialized, retrieved_docs
# ✅ GLOBAL variable
LAST_RETRIEVED_DOCS = []

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation for answering questions."""

    global LAST_RETRIEVED_DOCS

    # ✅ Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(query)

    # ✅ REMOVE duplicates
    unique_docs = []
    seen = set()

    for doc in retrieved_docs:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    retrieved_docs = unique_docs

    # ✅ Store globally (for returning later)
    LAST_RETRIEVED_DOCS = retrieved_docs

    # ✅ Limit content size (avoid token overflow)
    MAX_CHARS = 1000

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content[:MAX_CHARS]}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs



def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question

    Return:
        Dictionary containing:
          -answer: The generated answer
          -context: List of retrieved documents
    """
    # Prompt (modern replacement for system_prompt)
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "You are a helpful AI assistant that answers questions about LangChain documentation. "
    #      "You MUST use the retrieval tool before answering. "
    #      "Always cite sources. "
    #      "If answer is not found, say so."),
    #     ("user", "{input}"),
    #     ("placeholder", "{agent_scratchpad}")
    # ])
    
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )



    # base_prompt = hub.pull("hwchase17/react")

    # custom_prompt = base_prompt.partial(
    #     instructions="You are a helpful assistant for LangChain. Always cite sources. Use the tool before answering."
    # )

    from langchain_core.prompts import PromptTemplate

    # prompt = PromptTemplate.from_template("""
    # You are a helpful AI assistant that answers questions about LangChain documentation.

    # You have access to the following tools:
    # {tools}

    # Use the following format:

    # Question: {input}
    # Thought: think step-by-step about what to do
    # Action: one of [{tool_names}]
    # Action Input: the input to the tool
    # Observation: the result of the tool
    # ... (repeat as needed)
    # Thought: I now know the answer
    # Final Answer: provide the final answer and cite sources

    # IMPORTANT:
    # - Always use the retrieval tool before answering
    # - Always cite sources
    # - If the answer is not found, say so

    # {agent_scratchpad}
    # """)

    # prompt = PromptTemplate.from_template("""
    # You are a helpful AI assistant that answers questions about LangChain documentation.

    # You MUST follow this format EXACTLY:

    # Question: {input}
    # Thought: think step-by-step
    # Action: one of [{tool_names}]
    # Action Input: input to tool
    # Observation: result
    # ... repeat as needed ...
    # Thought: I now know the answer
    # Final Answer: your final answer with sources

    # RULES:
    # - ALWAYS use the tool first
    # - ALWAYS follow the format EXACTLY
    # - NEVER skip "Thought" or "Action"
    # - DO NOT jump directly to Final Answer

    # Available tools:
    # {tools}

    # {agent_scratchpad}
    # """)



    
    # prompt = PromptTemplate.from_template("""
    # Answer the question using the tool.

    # Format:
    # Question: {input}
    # Thought:
    # Action: [{tool_names}]
    # Action Input:
    # Observation:
    # ...
    # Final Answer:

    # Tools:
    # {tools}

    # {agent_scratchpad}
    # """)


    prompt = PromptTemplate.from_template("""
    Answer the question using the tool.

    Follow this format EXACTLY:

    Question: {input}
    Thought: think step-by-step
    Action: one of {tool_names}
    Action Input: the input to the tool
    Observation: the result of the tool
    ... repeat if needed ...
    Thought: I now know the answer
    Final Answer: provide the answer with sources

    Rules:
    - Do NOT use brackets in Action
    - Action must be exactly one of the tool names
    - Always use the tool before answering

    Tools:
    {tools}

    {agent_scratchpad}
    """)


    
  # ✅ Create agent
    agent = create_react_agent(
        model,
        tools=[retrieve_context],
        prompt=prompt
    )

    # ✅ MUST use executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[retrieve_context],
        verbose=True, handle_parsing_errors=True
    )

    # ✅ Run
    response = agent_executor.invoke({
        "input": query
    })

    # ✅ Extract answer
    answer = response["output"]

    # ✅ Extract retrieved documents
    context_docs = []
    for step in response.get("intermediate_steps", []):
        tool_message = step[1]

        if isinstance(tool_message, ToolMessage) and hasattr(tool_message, "artifact"):
            if isinstance(tool_message.artifact, list):
                context_docs.extend(tool_message.artifact)

    # return {
    #     "answer": answer,
    #     "context": context_docs
    # }
    
    return {
            "answer": answer,
            "context": LAST_RETRIEVED_DOCS
        }




    # agent = create_react_agent(model, tools=[retrieve_context], prompt=prompt)

    # #invoke the agent
    # messages = [{"role": "user", "content": query}]
    
    # # # Invoke the agent
    # # agent.invoke({"messages": messages})

    # # Invoke the agent
    # response = agent.invoke({"messages": messages})

    



    # # Extract the answer from the last AI message
    # answer = response["messages"][-1].content
    
    # # Extract context documents from ToolMessage artifacts
    # context_docs = []
    # for message in response["messages"]:
    #     # Check if this is a ToolMessage with artifact
    #     if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
    #         # The artifact should contain the list of Document objects
    #         if isinstance(message.artifact, list):
    #             context_docs.extend(message.artifact)
    # return {
    # "answer": answer,
    # "context": context_docs
    # }




if __name__ == '__main__':
    result = run_llm(query="What is RAG?")
    print(result) 
