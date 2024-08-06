from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.cache import InMemoryCache
from model import LanguageModelPipeline
from pymongo import MongoClient
from dotenv import load_dotenv
import gradio as gr
import google.generativeai as genai
import langchain
import torch
import os
import getpass
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json

#---- Enviroment Variables ----#
load_dotenv()
model_file_path = os.getenv('MODEL_FILE_PATH') or './models/ggml-vistral-7B-chat-q8.gguf'
model_embedding_name = os.getenv('MODEL_EMBEDDING_NAME') or 'bkai-foundation-models/vietnamese-bi-encoder'
vectorDB_path = os.getenv('VECTOR_DB_PATH') or './db'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = os.getenv('STORE_CACHE_PATH') or './cache/'
template = os.getenv('TEMPLATE')

#---- MongoDB Atlas ----#
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_ATLAS_CLUSTER_URI')
DB_NAME = os.getenv('DB_NAME') or 'langchain_db'
COLLECTION_NAME = os.getenv('COLLECTION_NAME') or 'vector_db'
ATLAS_VECTOR_SEARCH_INDEX_NAME  = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME') or 'vector_search_index'

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[DB_NAME][COLLECTION_NAME]
print("Kết nối thành công:", client[DB_NAME].name)


#---- Language Model Pipeline ----#
langchain.llm_cache = InMemoryCache()
pipeline = LanguageModelPipeline(
    model_file_path=model_file_path,
    model_embedding_name= model_embedding_name, 
    vectorDB_path=vectorDB_path,
    cache_path=store
)

llm = pipeline.load_model(
    model_type='llama',
    temperature=0.01,
    context_length=2048, 
    max_new_tokens=2048,
    gpu_layers=32,
    threads=4
)

prompt = pipeline.create_prompt(template=template)
embedding = pipeline.get_embedding()

#---- MongoDB Atlas Vector Search ----#
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

#---- FAISS Retriever ----#
# db = pipeline.load_db()
# llm_chain = pipeline.create_chain(llm, prompt, db, 3, True)

#---- Wiki Retriever ----#
# llm_chain = pipeline.create_chain_wiki(
#     llm=llm,
#     prompt=prompt,
#     top_k_documents=2,
#     return_source_documents=True
# )

#---- EnsembleRetriever with BM25 search and semantic search (Faiss or MongoDB)  ----#
llm_chain = pipeline.create_chain_ensemble(
    llm=llm, 
    prompt=prompt, 
    collection=collection,
    db=vector_store,
    top_k_documents=3,
    return_source_documents=True
)

#---- Gradio ----#
def respond(message, 
            history: list[tuple[str, str]], 
            system_message, 
            max_tokens, 
            temperature, 
            top_k_documents,
            ):
    
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": message})
    response = llm_chain.invoke({"query": message})

    yield response['result']

    
#---- Gradio Interface ----#
demo = gr.ChatInterface(
    respond,
    title="QA History",
    additional_inputs=[
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Top k documents to search for answers in",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()