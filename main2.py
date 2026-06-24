import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer

load_dotenv()
print("Initializing components.... ")

# embeddings = OpenAIEmbeddings()
# llm = ChatOpenAI()


# ✅ changed embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1"
)

# ✅ changed LLM
llm = ChatOllama(model="mymodel")  # or mistral, phi3, etc.


vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# prompt_template = ChatMessagePromptTemplate.from_template(
#     """Answer the question based only on the following context:
#     {context}
#     Question: {question}
#     provide a detailed answer:
#     """
# )

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    question: {question}
    provide a detailed answer:
    """
)

def format_docs(docs): 
    """Format retrieced documets into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_icel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them and generates a response.
    """
    #step 1: Retriever relevant documents
    docs = retriever.invoke(query)

    #Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted message
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
    
    Advantages over non-LCEL approach:
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain



if __name__=="__main__":
    print("Retrieving ...")

    #Query
    query = "what is Pinecone in machine learning?"

    # #======================================================
    # #Option 0: Row invocation without RAG
    # #================================================
    # print("\n"+"+" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # # print("anther part of the messa??ge is RAG")
    # print(result_raw.content)
    # print("End of raw message without RAG")





    # #======================================================
    # #Option 1: Use implementation WITHOUT LCEL
    # #================================================
    # print("\n"+"+" * 70)
    # print("IMPLEMENTATION 1: Without LCEL")
    # print("=" * 70)
    # result_witout_lcel = retrieval_chain_without_icel(query)
    # print("\nAnswer:")
    # # print("anther part of the messa??ge is RAG")
    # print(result_witout_lcel)
    # print("End of raw message RAG  without LCEL")



 #======================================================
    #Option 2: Implementation WITH LCEL (Better Approach)
    #================================================
    print("\n"+"+" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)
    print("why LCEL is better:")
    print("-more concise and declerative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvike()")
    print("- Easy to compose with other chains")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    # print("anther part of the messa??ge is RAG")
    print(result_with_lcel)
    print("End of raw message  RAG with LCEL")