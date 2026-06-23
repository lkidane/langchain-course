import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_community.embeddings import HuggingFaceEmbeddings

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

#configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1"
)

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tevily_extract = TavilyExtract()
tevily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

#log the documentation injection

async def main():
    """ Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        " TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    #Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth":5,
        "extract_depth": "advanced"
        #"instructions": "content on ai agents",
    })
    # res = tevily_map.invoke({
    # "url": "https://python.langchain.com/",

    # })



    # from tavily result we create langchain document
    #Convert Tavily crawl results to LangChain Document objects
    all_docs = [Document(page_content=result['raw_content'], metadata={"source":result['url']})  for result in res['results']]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )


    #Chunk the retrieved document
    # Split documents into chunks

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documets with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )



    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


    #Indexing the results

async def index_documents_async(documents: List[Document], batch_size: int=50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" VectorStore Indexing: preparing to add  {len(documents)} documets to vector store",
        Colors.DARKCYAN,
    )
    #create batches
    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(
        f" VectorStore Indexing: split into  {len(batches)} batches of {batch_size} documents each",
        Colors.DARKCYAN,
    )


    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: failed to add batch {batch_num} - {e}")
            return False
        return True
    # Process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
            






if __name__=="__main__":
    asyncio.run(main())