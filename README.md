# VietNamese Chatbot RAG with LLM: Ctransformers, Langchain, MongoDB or FAISS
___
## Environment variables:
```
HUGGINGFACE_TOKEN = ''
MONGODB_ATLAS_CLUSTER_URI = ''
MODEL_FILE_PATH = './models/.gguf'
MODEL_EMBEDDING_NAME = ''
VECTOR_DB_PATH = './db'
STORE_CACHE_PATH = './cache'
DB_NAME = ''
COLLECTION_NAME = ''
ATLAS_VECTOR_SEARCH_INDEX_NAME = ''
TEMPLATE = ''
```
___
## How to run:
1. Install library:`pip install -r requirements.txt`
2. Download file model `.gguf` to path: `models/`
3. Run cmd: `python app.py`
4. Model will running on local URL:  http://127.0.0.1:7860

*Note: You can upload file .pdf or .txt to `documents/` and run vector_db.py for custom data.*
___
## Models used:
- Embedding: 
    - [bkai-foundation-models/vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder)
- LLMs: 
    - [Vistral-7B-Chat-GGUF](https://huggingface.co/uonlp/Vistral-7B-Chat-gguf)
    - [VinaLLaMA-2.7B-Chat-GGUF](https://huggingface.co/vilm/vinallama-2.7b-chat-GGUF)
___

## Retriever used:
- Faiss Retriver
- Mongodb Retriever
- Ensemble Retriever

## Demo
<!-- ![Demo IMG](images/image.png) -->
![Demo](images/image1.png)
<br>
![Demo](images/image2.png)
<br>
![Demo](images/image3.png)
<br>
![Demo](images/image4.png)
<br>
![Demo](images/image5.png)
___
