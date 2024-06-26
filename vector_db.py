from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

class DocumentsFaissDB:
    def __init__(self, db_path, model_name, documents_path, chunk_size, chunk_overlap) -> None:
        self.db_path = db_path
        self.documents_path = documents_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = self.load_embeddings()

    def load_documents(self):
        pdf_loader = DirectoryLoader(self.documents_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = pdf_loader.load()
        return documents
    
    def text_splitter(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def load_embeddings(self):
        embeddings = GPT4AllEmbeddings(
            model_name=self.model_name,
            gpt4all_kwargs={
                'allow_download': True,
            }
        )
        return embeddings
    
    def create_db(self):
        # Load documents and split them into chunks
        documents = self.load_documents()
        chunks = self.text_splitter(documents)

        # Create embeddings
        self.embeddings = self.load_embeddings()

        # Create db
        db = FAISS.from_documents(chunks, self.embeddings)
        db.save_local(self.db_path)

        return db

    def load_db(self):
        self.embeddings = self.load_embeddings()
        db = FAISS.load_local(
            self.db_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        return db

# pdf_data_path = './documents'
# vector_db_path = './db'
# model_name = 'all-MiniLM-L6-v2.gguf2.f16.gguf'

# def create_db_from_files():
#     # Khai bao loader de quet toan bo thu muc dataa
#     loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
#     chunks = text_splitter.split_documents(documents)

#     # Embeding
#     embeddings = GPT4AllEmbeddings(
#             model_name=model_name,
#             gpt4all_kwargs={
#                 'allow_download': True,
#             }
#         )
#     db = FAISS.from_documents(chunks, embeddings)
#     db.save_local(vector_db_path)
#     return db


# create_db_from_files()