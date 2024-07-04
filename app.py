import gradio as gr
from huggingface_hub import InferenceClient
import langchain
import os
from langchain_community.llms import CTransformers
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.cache import InMemoryCache
from langchain.embeddings import CacheBackedEmbeddings

import torch

model_name = "vilm/vinallama-2.7b-chat-GGUF"
model_file_path = './models/vinallama-7b-chat_q5_0.gguf'
model_embedding_name = 'bkai-foundation-models/vietnamese-bi-encoder'
vectorDB_path = './db'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = LocalFileStore("./cache/")

def load_model(model_file_path, 
               model_type, 
               temperature=0.01,
               context_length=1024,
               max_new_tokens=1024
               ):
    llm = CTransformers(
        model = model_file_path,
        model_type = model_type,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        config = {
            'context_length': context_length,
        },
    )
    return llm

def create_embedding_model():
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_embedding_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(embeddings,
                                                      store,
                                                      namespace=model_embedding_name)
    return embedder

embeddings = create_embedding_model()

def load_db():
    embeddings
    db = FAISS.load_local(vectorDB_path, embeddings, allow_dangerous_deserialization=True)
    return db

def create_prompt(template):
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'],
    )

    return prompt

def create_chain(llm, 
                prompt, 
                db, 
                top_k_documents=3, 
                return_source_documents=True):
    print(prompt)
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = db.as_retriever( 
            search_kwargs={
                "k": top_k_documents
                }
            ),
        return_source_documents = return_source_documents,
        chain_type_kwargs = {
            'prompt': prompt,
        },
    )

    return chain

db = load_db()
llm = load_model(
    model_file_path=model_file_path, 
    model_type='llama',
    context_length=2048,
    max_new_tokens=2048,
    )


template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời. \n
{context}<|im_end|>\n
<|im_start|>user\n
{question}!<|im_end|>\n
<|im_start|>
Hãy đảm bảo rằng bạn cung cấp câu trả lời với các mốc thời gian chính xác nhất có thể.
assistant
"""

prompt = create_prompt(template=template)
llm_chain = create_chain(llm, prompt, db)

langchain.llm_cache = InMemoryCache()

def respond(message, 
            history: list[tuple[str, str]], 
            system_message, 
            max_tokens, 
            temperature, 
            top_k_documents,
            ):
    
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = llm_chain.invoke({"query": message})
    
    # # In ra các context thu được
    # source_documents = response.get('source_documents', [])
    # with open('res.txt', 'w', encoding='utf-8') as f:
    #     for doc in source_documents:
    #         f.write(doc.page_content + "\n\n")

    yield response['result']

    

demo = gr.ChatInterface(
    respond,
    title="Chatbot",
    additional_inputs=[
        # gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Top k documents to search for answers in",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()