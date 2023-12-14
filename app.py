from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from pydantic import BaseModel
import torch
from langchain.llms import HuggingFacePipeline

app = FastAPI()

class InputData(BaseModel):
    prompt: str

class OutputData(BaseModel):
    prompt: str

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, cache_dir='models/')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='models/')

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    # Langchain needs full text
    return_full_text=True,
    task="text-generation",
    temperature=0.0,
    max_new_tokens=512,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

@app.post("/generate", response_model=OutputData)
def generate(request: Request, input_data: InputData):
    prompt = input_data.prompt
    
    response = model(prompt)[0]["generated_text"]
    
    return OutputData(response=response)