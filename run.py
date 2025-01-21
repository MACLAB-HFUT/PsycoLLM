import os
import json
import torch
import logging
import deepspeed
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import (login, HfFolder, snapshot_download,)

"""
    快速开始：
    deepspeed --num_gpus=2 run.py
"""

"""
    初始化聊天接口参数说明：
    model_name: HuggingFace上的模型名称，默认为MACLAB-HFUT/PsycoLLM1.5
    model_path: 本地模型路径（可选）
    device: 运行设备，默认使用CUDA（如果可用）
    max_new_tokens: 生成的最大新token数量
    temperature: 温度参数，控制生成的随机性
    top_p: 用于核采样的概率阈值
    use_auth_token: HuggingFace的访问令牌（可选）
    cache_dir: 模型缓存目录（可选）
    local_rank: 分布式训练中的本地进程编号（可选）
"""
class PsycoLLMChat:
    def __init__(
        self,
        model_name: str = "MACLAB-HFUT/PsycoLLM",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_rank: int = -1
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'huggingface'
        self.local_rank = local_rank
        
        # 初始化对话历史，使用消息格式存储
        self.messages: List[Dict[str, str]] = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._setup_auth(use_auth_token)
        self._setup_distributed()
        self._load_model_and_tokenizer()
    
    def _setup_distributed(self):
        # 初始化分布式环境
        if self.local_rank == -1:
            self.world_size = 1
            return

        deepspeed.init_distributed()
        torch.cuda.set_device(self.local_rank)
        self.world_size = torch.distributed.get_world_size()
        self.logger.info(f"分布式环境设置完成。本地进程编号: {self.local_rank}, 总进程数: {self.world_size}")
    
    def _setup_auth(self, use_auth_token: Optional[str]):
        # 设置HuggingFace认证
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
                "未提供HuggingFace认证token。如需访问私有模型，请提供token。"
            )
            self.auth_token = None
    
    def _load_model_and_tokenizer(self):

        try:
            if self.model_path and os.path.exists(self.model_path):
                self.logger.info(f"正在从本地路径加载模型: {self.model_path}...")
                model_path = self.model_path
            else:
                self.logger.info(f"正在从HuggingFace下载模型 {self.model_name}...")
                model_path = snapshot_download(
                    repo_id=self.model_name,
                    token=self.auth_token,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=self.auth_token if not self.model_path else None
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=self.auth_token if not self.model_path else None
            )
            
            model.eval()
            
            # DeepSpeed推理配置
            ds_inference_config = {
                "tensor_parallel": {
                    "tp_size": self.world_size
                },
                "dtype": "fp16",
                "replace_method": "auto",
                "replace_with_kernel_inject": True
            }
            
            self.logger.info(f"当前GPU: {self.local_rank}, 总进程数: {self.world_size}")
            self.logger.info(f"DeepSpeed配置: {ds_inference_config}")
            
            self.model = deepspeed.init_inference(
                model=model,
                config=ds_inference_config
            )
            
            self.logger.info(f"模型加载成功！当前进程编号: {self.local_rank}, 总进程数: {self.world_size}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        # 更新消息历史
        if system_prompt and not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": prompt})
        
        try:
            # 使用消息模板格式化对话历史
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 对完整上下文进行分词
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            
            # 将输入移动到对应设备
            if self.local_rank != -1:
                input_ids = input_ids.to(f"cuda:{self.local_rank}")
            else:
                input_ids = input_ids.to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.module.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # 提取新生成的token（回复内容）
            new_tokens = outputs[0][len(input_ids[0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 将助手的回复添加到消息历史
            self.messages.append({"role": "assistant", "content": response})
            
            return {"response": response}

        except Exception as e:
            self.logger.error(f"生成回复时发生错误: {str(e)}")
            return {"error": str(e)}
    
    def chat(self, system_prompt: Optional[str] = None):
        # 交互式聊天接口
        if self.local_rank in [-1, 0]:
            print("欢迎使用PsycoLLM心理健康对话大模型！输入'quit'或'exit'可结束对话。")
            if system_prompt:
                print(f"\n系统提示词: {system_prompt}\n")
            
            while True:
                user_input = input("\n用户: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    print("\n感谢使用PsycoLLM，祝您生活愉快，再见！")
                    break
                    
                try:
                    response = self.generate_response(
                        user_input,
                        system_prompt=system_prompt if not self.messages else None
                    )
                    if "error" in response:
                        print(f"\n错误: {response['error']}")
                    else:
                        print("\n助手:", response["response"])
                except Exception as e:
                    print(f"\n错误: {str(e)}")
                    continue

    def clear_history(self):
        # 清除对话历史
        self.messages = []
        self.logger.info("对话历史已清除")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        chat_bot = PsycoLLMChat(
            model_path="MACLAB-HFUT/PsycoLLM",
            cache_dir="./model_cache",
            local_rank=args.local_rank
        )
        
        system_prompt = "角色：你是一名优秀的心理咨询助手，具有丰富的咨询经验和工作经验。你性格乐观开朗、热情待人；你逻辑清晰、善于倾听，具有强烈的同理心和同情心；你熟悉心理咨询的流程，遵循心理咨询的伦理，希望帮助你的用户提振心情、走出困境。\n任务：1、请你认真倾听用户的困扰和情感诉求，并表现出理解与共情。\n2、熟悉问询技巧，使用引导性问题，帮助用户深入思考和表达自己。\n3、使用积极正面的语言，让用户舒缓情绪，并提供实用且可行的建议。\n4、遵循心理咨询的伦理。"
        
        chat_bot.chat(system_prompt=system_prompt)
        
    except Exception as e:
        print(f"启动PsycoLLM时发生错误: {str(e)}")

if __name__ == "__main__":
    main()
