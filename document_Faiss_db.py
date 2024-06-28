from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


pdf_data_path = './documents'
vector_db_path = './db'
# model_name = 'all-MiniLM-L6-v2.gguf2.f16.gguf'
model_name = 'bkai-foundation-models/vietnamese-bi-encoder'


def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, 
        chunk_overlap=128
    )
    chunks = text_splitter.split_documents(documents)

    # embeddings = GPT4AllEmbeddings(
    #         model_name=model_name,
    #         gpt4all_kwargs={
    #             'allow_download': True,
    #         }
    #     )

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    return db

create_db_from_files()