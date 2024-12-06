import os
import json
import torch
import logging
import deepspeed
from pathlib import Path
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import (login, HfFolder, snapshot_download,)

"""
    快速开始：

    deepspeed --num_gpus=2 run.py --deepspeed_config ds_z3_config.json （使用deepspeed zero3）
    或
    python run.py

"""

"""
    初始化聊天接口
    
    model_name: HuggingFace上的模型名称, 默认为空
    max_length: 生成文本的最大长度
    temperature: 温度参数，控制生成的随机性
    top_p: 样本生成时的top-p采样参数
    use_auth_token: HuggingFace的访问令牌（可选）
    cache_dir: 模型缓存目录（可选）
    deepspeed_config: DeepSpeed配置文件路径（可选）
    local_rank: 当前进程的局部等级，用于分布式训练（可选）

"""
class PsycoLLMChat:
    def __init__(
        self,
        model_name: str = "MACLAB-HFUT/PsycoLLM",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        deepspeed_config: Optional[str] = None,
        local_rank: int = -1
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'huggingface'
        self.local_rank = local_rank
        self.deepspeed_config = deepspeed_config
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._setup_auth(use_auth_token)
        
        if self.deepspeed_config:
            deepspeed.init_distributed()
        
        self._load_model_and_tokenizer()
    
    
    def _setup_auth(self, use_auth_token: Optional[str]):
        if use_auth_token:
            login(token=use_auth_token)
            self.auth_token = use_auth_token
        elif os.getenv("HUGGINGFACE_TOKEN"):
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
            self.auth_token = os.getenv("HUGGINGFACE_TOKEN")
        elif HfFolder.get_token():
            self.auth_token = HfFolder.get_token()
        else:
            self.logger.warning(
                "您未提供HuggingFace的认证token。如果您需要访问私有模型，请提供token。"
            )
            self.auth_token = None
    
    
    def _load_model_and_tokenizer(self):
        self.logger.info(f"正在从HuggingFace上加载模型{self.model_name}，请稍等...")
        
        try:
            model_path = snapshot_download(
                repo_id=self.model_name,
                token=self.auth_token,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=self.auth_token
            )
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=self.auth_token
            )

            if self.deepspeed_config:
                with open(self.deepspeed_config) as f:
                    ds_config = json.load(f)
                
                # 确保配置中包含了必要参数
                if "train_micro_batch_size_per_gpu" not in ds_config:
                    ds_config["train_micro_batch_size_per_gpu"] = 1
                if "train_batch_size" not in ds_config:
                    world_size = int(os.getenv("WORLD_SIZE", "1"))
                    ds_config["train_batch_size"] = ds_config["train_micro_batch_size_per_gpu"] * world_size

                self.model = deepspeed.init_inference(
                    model=model,
                    mp_size=1,  # 模型并行度设置为1
                    dtype=torch.float16,
                    replace_method='auto',
                    replace_with_kernel_inject=True
                )
            else:
                self.model = model.to(self.device)
            
            self.logger.info("模型加载成功！")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
        

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        if system_prompt:
            input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        if self.deepspeed_config:
            inputs = {k: v.to(self.model.module.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                if self.deepspeed_config:
                    outputs = self.model.module.generate(
                        **inputs,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **kwargs
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **kwargs
                    )
            
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

             # 从响应中提取心理咨询助手的回复
            response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
            return {"response": response}

        except Exception as e:
            self.logger.error(f"生成响应时发生错误: {str(e)}")
            return {"error": str(e)}
    

    def chat(self, system_prompt: Optional[str] = None):
        if self.local_rank in [-1, 0]:
            print("欢迎使用PsycoLLM中文心理健康对话模型！输入'quit'或'exit'可结束对话。")
            if system_prompt:
                print(f"\n您的系统提示词为： {system_prompt}\n")
            
        while True:
            if self.local_rank in [-1, 0]:
                user_input = input("\n用户（您）: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    print("\n感谢您的使用，祝您生活愉快，再见！")
                    break
                    
                try:
                    response = self.generate_response(
                        user_input,
                        system_prompt=system_prompt
                    )
                    if "error" in response:
                        print(f"\n发生错误: {response['error']}")
                    else:
                        print("\n咨询助手:", response["response"])

                except Exception as e:
                    print(f"\n发生错误: {str(e)}")
                    continue

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    
    try:
        chat_bot = PsycoLLMChat(
            model_name="MACLAB-HFUT/PsycoLLM",
            use_auth_token=hf_token,
            cache_dir="./model_cache",
            deepspeed_config=args.deepspeed_config,
            local_rank=args.local_rank
        )
        
        system_prompt = "角色：你是一名优秀的心理咨询助手，具有丰富的心理咨询经验和工作经验。你性格乐观开朗、热情待人；你逻辑清晰、善于倾听，具有强烈的同理心和同情心；你熟悉心理咨询的流程，遵循心理咨询的伦理，希望帮你的用户提振心情、走出困境。\n任务：1、请你认真倾听用户的困扰和情感诉求，并表现出理解与共情。\n2、熟悉问询技巧，使用引导性问题，帮助用户深入思考和表达自己。\n3、使用积极正面的语言，让用户舒缓情绪，并提供实用且可行的建议。"
        chat_bot.chat(system_prompt=system_prompt)
        
    except Exception as e:
        print(f"启动PsycoLLM时发生错误: {str(e)}")


if __name__ == "__main__":
    main()
