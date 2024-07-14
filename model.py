from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import torch

class LanguageModelPipeline:
    def __init__(self, model_file_path, model_embedding_name, vectorDB_path, cache_path='./cache/'):
        self.model_file_path = model_file_path
        self.model_embedding_name = model_embedding_name
        self.vectorDB_path = vectorDB_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.store = LocalFileStore(cache_path)
        self.embeddings = self.create_embedding_model()

    def get_embedding(self):
        return self.embeddings

    def load_model(self, model_type, temperature=0.01, context_length=1024, max_new_tokens=1024, gpu_layers=10, threads=4):
        llm = CTransformers(
            model=self.model_file_path,
            model_type=model_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            config={
                'gpu_layers': gpu_layers,
                'context_length': context_length,
                'threads': threads,
            },
        )
        return llm

    def create_embedding_model(self):
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_embedding_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, self.store, namespace=self.model_embedding_name)
        return embedder

    def load_db(self):
        db = FAISS.load_local(self.vectorDB_path, self.embeddings, allow_dangerous_deserialization=True)
        return db

    def create_prompt(self, template):
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question'],
        )
        return prompt

    def create_chain(self, llm, prompt, db, top_k_documents=3, return_source_documents=True):
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(
                search_kwargs={
                    "k": top_k_documents
                }
            ),
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                'prompt': prompt,
                'document_variable_name': 'context',  
            },
        )
        return chain
    
    def create_chain_hybird(self, llm, prompt, collection, db, top_k_documents=3, return_source_documents=True):
        docs = collection.find()
        documents = [
            Document(
                page_content=doc['text'], 
                metadata={
                    'page': doc['page'] if (doc.get('page')) else 1,
                    'source': doc['source'],
                    'source_type': doc['source_type']
                }
            ) for doc in docs
        ]

        bm25_retriever = BM25Retriever.from_documents(documents)
        semantic_retriever = db.as_retriever(search_kwargs={"k": top_k_documents})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], 
            weights=[0.2, 0.8]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=ensemble_retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                'prompt': prompt,
                'document_variable_name': 'context',  
            },
        )

        return chain