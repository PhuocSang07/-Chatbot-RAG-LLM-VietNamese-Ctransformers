from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from vector_db import DocumentsFaissDB


vectorDB_path = './db'
documents_path = './documents'
model_name = 'all-MiniLM-L6-v2.gguf2.f16.gguf'
# model_file_path = './models/vinallama-7b-chat_q5_0.gguf'
model_file_path = './models/ggml-vistral-7B-chat-q5_1.gguf'

def load_model():
    llm = CTransformers(
        model = model_file_path,
        model_type = 'llama',
        max_new_tokens = 1024,
        temperature = 0.01,
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
        chain_type_kwargs = {'prompt': prompt},
    )

    return chain



documentsFaissDB = DocumentsFaissDB(
    db_path=vectorDB_path, 
    model_name=model_name,
    documents_path=documents_path,
    chunk_size=1024,
    chunk_overlap=128
)

db = documentsFaissDB.create_db()
# db = documentsFaissDB.load_db()

llm = load_model()

# template = """<|im_start|>system
# Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời \n
# {context}<|im_end|>\n
# <|im_start|>user\n
# {question}!<|im_end|>\n
# <|im_start|>assistant
# """
template ="""<s>[INST] <<SYS>>
Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời
{context}
<</SYS>>

{question} [/INST]
"""

prompt = create_prompt(template=template)
llm_chain = create_chain(llm, prompt, db)

# Test the chain
question = ''
print("Nhập câu hỏi: ")
while (True):
    question = input()
    if(question == 'exit'):
        break
    response = llm_chain.invoke({"query": question})

    print("Câu hỏi: ",response['query'])
    print("Trả lời: ",response['result'])
    print("--",response['source_documents'][0])
