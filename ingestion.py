import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeSparseVectorStore, PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import ssl
import certifi
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    # ssl._create_default_https_context = ssl._create_unverified_context
    session = requests.Session()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    # pinecone is vector database
    #persistent storage, ability to search in the vector space, add new vectors to vector space
    # print(os.environ['PINECONE_API_KEY'])
    loader = TextLoader("C:/Users/lkidane/langchain-course/mediumblog1.txt", autodetect_encoding=True, encoding="utf-8")
    #Document loaders 
    document = loader.load()
    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created{len(texts)} chunks")

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # embeddings = OllamaEmbeddings(model="mymodel")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1"
    )

    print("Ingesting....")
    # vectorstore = FAISS.from_documents(texts, embeddings)
    
    # PineconeSparseVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ['INDEX_NAME']
    )

    print("done loading the embeddings...")
