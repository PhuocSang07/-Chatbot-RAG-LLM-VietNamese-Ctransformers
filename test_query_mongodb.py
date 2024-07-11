from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
import torch
import os


load_dotenv()
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')
DB_NAME = 'langchain_db'
model_embedding_name = os.getenv('MODEL_EMBEDDING_NAME') 
COLLECTION_NAME = 'vector_db'
ATLAS_VECTOR_SEARCH_INDEX_NAME  = 'vector_search_index'
AI21_TOKEN = os.getenv('AI21_TOKEN')
os.environ["AI21_API_KEY"] = AI21_TOKEN
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[DB_NAME][COLLECTION_NAME]

db = client[DB_NAME]
print("Kết nối thành công:", db.name)

collection = db[COLLECTION_NAME]

def create_embedding_model():
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_embedding_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, "./cache", namespace=model_embedding_name)
    return embedder

embeddings = create_embedding_model()

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

query = "Chiến dịch việt bắc thu đông"
query_embed = embeddings.embed_query(query)

semantic_search_results = collection.aggregate([
    {
        "$vectorSearch": {
            "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embed,
            "numCandidates": 200,
            "limit": 10
        }
    }
])


for i in semantic_search_results:
    print(i['text'])
    print('--------------------------------')

# res = compression_retriever.get_relevant_documents(query=query)
# print(res)

# for result in results:
    # print(result.page_content)
    # print(result)
    # print('--------------')
