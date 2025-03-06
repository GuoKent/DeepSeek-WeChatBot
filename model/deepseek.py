import os
import sys
from pathlib import Path
from itertools import chain
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import time
import torch
import yaml
from utils.utils import load_sys_prompt, load_model_config
from wcferry import Wcf
from typing import Optional
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

_model_root = "C:\Projects\DeepSeek"
_model_options = (
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
)

class DeepSeek:
    def __init__(self, config: dict, ):
        self.config = config
        self.mode = config.get('mode', 'local')  # options: {local, api}
        self.frame = config.get('frame', 'transformers')  # options: {transformers, vllm}
        assert self.mode in ('local', 'api'), f"config.mode shoule be in ('local', 'api'), but got {self.mode}"
        assert self.frame in ('transformers', 'vllm'), f"config.frame shoule be in ('transformers', 'vllm'), but got {self.frame}"

        # 模型扮演的角色，系统prompt
        self.role = config.get('role', 'default')
        self.sys_prompt = load_sys_prompt(self.role)
        self.default_messages = [{"role": "system", "content": self.sys_prompt}]
        # 历史记录长度
        self.history = config.get('history', 3)
        # 初始化模型
        self._initialize_local(config) if self.mode == 'local' else self._initialize_api(config)
    
    def _initialize_local(self, config):
        '''
            本地部署初始化
        '''
        if self.frame == "transformers":
            model_name = config.get('model_name', None)
            model_path = os.path.join(_model_root, model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cuda:0",
                # llm_int8_enable_fp32_cpu_offload=True,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                attn_implementation="flash_attention_2",
                quantization_config={
                    "load_in_4bit": True,  # 4-bit量化
                    "bnb_4bit_compute_dtype": torch.float16
                }
            )
            self.model = torch.compile(self.model)  # dynamic graph compile (improve not much)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.streamer = TextStreamer(self.tokenizer)
        else:
            # TODO add vllm frame, on Mac or Linux
            model = LLM(model=model_name, tensor_parallel_size=1)
            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0
            )
        self.generate_mode = self.frame
    
    def _initialize_api(self, config):
        '''
            使用api初始化
        '''
        # TODO
        self.generate_mode = "api"
        pass

    def clean_history_messages(self, messages: list, history: int=3) -> list:
        '''
        清除历史记录
        Args:
            messages: list, {{"role": "system", "content": self.sys_prompt}, ...}
        '''
        if history == 0:
            return [{"role": "system", "content": self.sys_prompt}]
        elif len(messages) <= 2 * history + 1:
            return messages
        else:
            return list(chain([{"role": "system", "content": self.sys_prompt}], messages[-2 * history:]))

    def generate_tfs(self, messages: list):
        end = time.time()
        # self.messages.append({"role": "user", "content": input_txt})
        # self.clean_history_messages(self.history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.amp.autocast("cuda", enabled=True):
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                # load_in_4bit=True,                        # 4-bit量化
                num_beams=1,                                # 禁用束搜索
                do_sample=True,                            # 禁用采样
                use_cache=True,                             # 启用KV缓存
                pad_token_id=self.tokenizer.pad_token_id,   # 配置padding_id
                eos_token_id=self.tokenizer.eos_token_id,   # 配置eos_id
                streamer = self.streamer,                   # 流式输出
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            token_num = len(generated_ids[0])
            cost_time = time.time() - end
            token_speed = token_num / cost_time

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # self.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "token_num": token_num,
            "token_speed": token_speed,
            "cost_time": cost_time,
        }

    def generate_vllm(self, messages: list):
        # TODO
        pass
    
    def generate_api(self, messages: list):
        # TODO
        pass

    def generate(self, messages: list):
        generate_mode = {
            "transformers": self.generate_tfs,
            "vllm": self.generate_vllm,
            "api": self.generate_api,
        }
        return generate_mode[self.generate_mode](messages)

if __name__ == "__main__":
    configs = load_model_config(model="deepseek")
    print(configs)
    deepseek = DeepSeek(configs)
    while True:
        input_txt = input(">>>")
        outputs = deepseek.generate(input_txt)
        # print(outputs["response"])
        print(f"Cost time: {outputs['cost_time']:.2f}s | Token nums: {outputs['token_num']} | Token speed: {outputs['token_speed']:.2f} token/s")