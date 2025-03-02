# -*- coding: utf-8 -*-
import os
import torch
import time
from vllm import LLM, SamplingParams
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['USE_FLASH_ATTENTION'] = "1"
# os.environ['NVIDIA_TF32_OVERRIDE'] = "1"
torch.cuda.CUDAGraph()

model_options = (
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B"
)
model_name = "../DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    # use_flash_attention_2=True,
    attn_implementation="flash_attention_2",
)
model = torch.compile(model)  # 动态图编译(提升效果不明显)

# model = LLM(model=model_name, tensor_parallel_size=1)
# sampling_params = SamplingParams(
#     max_tokens=512,
#     temperature=0
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)

print(model.device)
print(model.config)

messages = [
        {"role": "system", "content": "你是DeepSeek-R1，一个非常有用的AI助手。"},
    ]

while True:
    prompt = input(">>>")
    if prompt == '/exit':
        break
    elif prompt == '/clean':
        messages = []
        continue
    end = time.time()
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.amp.autocast("cuda", enabled=True):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            # load_in_4bit=True,   # 4-bit量化
            num_beams=1,         # 禁用束搜索
            do_sample=False,     # 禁用采样
            use_cache=True,       # 启用KV缓存
            pad_token_id=tokenizer.pad_token_id,  # 配置padding_id
            eos_token_id=tokenizer.eos_token_id,  # 配置eos_id
            streamer = streamer,  # 流式输出
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]  # 模型生成输出包括 问题+输出，需要把问题从最终结果中去除
    token_num = len(generated_ids[0])
    cost_time = time.time() - end
    token_speed = token_num / cost_time

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    messages.append({"role": "assistant", "content": response})
    print(response)
    print(f"Cost time: {cost_time:.2f}s | Token nums: {token_num} | Token speed: {token_speed:.2f} token/s")