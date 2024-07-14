from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.cache import InMemoryCache
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema import Document
from pymongo import MongoClient
from dotenv import load_dotenv
import langchain
import torch
import re
import os


load_dotenv()
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')


DB_NAME = 'langchain_db'
COLLECTION_NAME = 'vector_db'
ATLAS_VECTOR_SEARCH_INDEX_NAME  = 'vector_search_index'
AI21_TOKEN = os.getenv('AI21_TOKEN')
os.environ["AI21_API_KEY"] = AI21_TOKEN
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
model_embedding_name = os.getenv('MODEL_EMBEDDING_NAME') or 'bkai-foundation-models/vietnamese-bi-encoder'

def create_embedding_model():
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_embedding_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, LocalFileStore("./cache/"), namespace=model_embedding_name)
    return embedder

langchain.llm_cache = InMemoryCache()
embeddings = create_embedding_model()

def clean_text(text):
    text = re.sub(r'[^\w\s,.-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(" \n", "\n").replace("\n ", "\n").replace("\n", "\n\n")
    
    return text

def create_chunks_from_files(pdf_data_path):
    text_loader_kwargs={'autodetect_encoding': True}
    pdf_loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls = PyPDFLoader)
    txt_loader = DirectoryLoader(pdf_data_path, glob="**/*.txt", loader_cls = TextLoader, loader_kwargs=text_loader_kwargs)

    documents = pdf_loader.load() + txt_loader.load()


    text_splitter = AI21SemanticTextSplitter(chunk_size=4096, chunk_overlap=1024)

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    return chunks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

chunks = create_chunks_from_files(pdf_data_path='./documents',)

vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)