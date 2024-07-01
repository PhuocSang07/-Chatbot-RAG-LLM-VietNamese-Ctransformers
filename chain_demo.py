from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "vilm/vinallama-2.7b-chat-GGUF"
model_file_path = './models/vinallama-7b-chat_q5_0.gguf'
model_embedding_name = 'bkai-foundation-models/vietnamese-bi-encoder'

def load_model():
    llm = CTransformers(
        model = model_file_path,
        model_type = 'llama',
        max_new_tokens = 1024,
        temperature = 0.01,
        config = {
            'context_length': 1024,
        },
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'],
    )

    return prompt

def create_chain(llm, prompt, db):
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = db.as_retriever( search_kwargs={"k": 3}),
        return_source_documents = True,
        chain_type_kwargs = {
            'prompt': prompt,
        },
    )

    return chain

vectorDB_path = './db'
def load_db():
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_embedding_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.load_local(vectorDB_path, embeddings, allow_dangerous_deserialization=True)
    return db


db = load_db()
llm = load_model()

template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời \n
{context}<|im_end|>\n
<|im_start|>user\n
{question}!<|im_end|>\n
<|im_start|>assistant
"""

prompt = create_prompt(template=template)
llm_chain = create_chain(llm, prompt, db)

# Test the chain
question = "Diễn biến Chiến dịch biên giới thu đông 1950"
response = llm_chain.invoke({"query": question})
print(response)
print()

print(response['query'])
print(response['result'])
print()

print(response['source_documents'])
