import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from langchain.llms import HuggingFacePipeline
from loguru import logger


def quantize_config():
    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info(f'Your GPU supports bfloat16: accelerate training with bf16=True')
    return quantization_config


def load_llm_langchain(model_name: str, temperature: float = .2, max_new_tokens: int = 1000):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantize_config(),
    )

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        temperature=temperature,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=max_new_tokens,
    )

    model = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return model


def messages_to_prompt(messages):
    prompt = ''
    for message in messages:
        if message.role == 'system':
            prompt += f'<|system|>\n{message.content}\n'
        elif message.role == 'user':
            prompt += f'<|user|>\n{message.content}\n'
        elif message.role == 'assistant':
            prompt += f'<|assistant|>\n{message.content}\n'

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith('<|system|>\n'):
        prompt = '<|system|>\n\n' + prompt

    # add final assistant prompt
    prompt = prompt + '<|assistant|>\n'
    return prompt


def load_llm_idx():
    return HuggingFaceLLM(
        model_name='HuggingFaceH4/zephyr-7b-alpha',
        tokenizer_name='HuggingFaceH4/zephyr-7b-alpha',
        query_wrapper_prompt=PromptTemplate('<|system|>\n\n<|user|>\n{query_str}\n<|assistant|>\n'),
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={'quantization_config': quantize_config()},
        # tokenizer_kwargs={},
        generate_kwargs={'temperature': 0.7, 'top_k': 50, 'top_p': 0.95, 'do_sample': True},
        messages_to_prompt=messages_to_prompt,
        device_map='auto',
    )
