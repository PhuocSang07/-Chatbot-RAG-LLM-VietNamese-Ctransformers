from langchain_community.llms import CTransformers, HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, BitsAndBytesConfig
import torch
import os

model_path = "vinai/PhoGPT-7B5-Instruct"
token="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"   
# model_path = "vilm/vinallama-7b"
# model_path="vilm/vinallama-2.7b-chat"

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    token=token
)

# Set config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    token=token
)
config.init_device = device
config.temperature = 0.01
# config.max_length =300
# config.eos_token_id=tokenizer.eos_token_id
# config.pad_token_id=tokenizer.pad_token_id
# config.do_sample = True

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    config=config,
    trust_remote_code=True,
    token=token
)

model.eval()

text_generation_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer, 
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    return_full_text=True,
    max_new_tokens=300,
)

prompt = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"

input_prompt = prompt.format_map(
    {'instruction': 'Hãy viết một đoạn văn bản về một chủ đề bất kỳ bạn muốn.'}
)

my_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
my_pipeline(input_prompt)