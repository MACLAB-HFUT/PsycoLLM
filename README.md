[//]: # (![# PsycoLLM]&#40;assets/logo.png&#41;)

# 中文心理大模型PsycoLLM
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a><img src="https://img.shields.io/badge/python-3.8+-blue.svg" /><a href='https://arxiv.org/pdf/2407.05721'><img src='https://img.shields.io/badge/ArXiv-2407.05721v2-red'></a>

Paper here -> PsycoLLM: [Enhancing LLM for Psychological Understanding and Evaluation](https://arxiv.org/pdf/2407.05721)

## 最近更新
-[2024.11.4]  PsycoLLM: Enhancing LLM for Psychological Understanding and Evaluation 被 IEEE Transactions on Computational Social Systems 接收

-[2024.9.22] 我们的中文心理大模型PsycoLLM正式发布！如有需要下载模型，请点击此处：[MACLAB-HFUT/PsycoLLM](https://huggingface.co/MACLAB-HFUT/PsycoLLM)

## 项目简介

### - 背景概述
心理健康问题一直备受社会关注。在当代社会，每个人都可能遇上或多或少的难题，在心底慢慢堆积，便可能形成大的心理健康问题。青少年可能是因为校园霸凌、同学相处、原生家庭，年轻人可能是学业繁忙、工作压力大，老年人则可能是缺乏陪伴，需要抒发他们内心的不安与不满，以缓解环绕内心已久的心理问题。为此，我们精心制作了一份数据集，基于此进行大模型的微调，拟用于心理咨询。微调后的大模型在我们构建的评估集上表现优异，其 MCQ 准确率超过60%。

### - 以下是PsycoLLM的数据集准备过程的概览：
![Overview of dataset preparation.](assets/dataset_overview.jpg)

### - 以下是一个完整的使用pipeline生成多轮对话的过程，其中包括数据生成、论证支持以及优化阶段：
![Examples of the generated multi-turn dialogue data.](assets/multi_turn_example_v2.jpg)

### - 而生成基于知识的QA问答对的过程则是：
![Knowledge-based QA generation.](assets/knowledge-base-QA.jpg)

## 亮点

- 我们根据官方的全国心理咨询师考试制作了一个benchmark。

- PsycoLLM 在该benchmark中的平均 MCQ 准确率超过了 60%。

## 快速使用
1.克隆本项目至本地
```bash
git clone https://github.com/MACLAB-HFUT/PsycoLLM.git
```
2.配置环境
```bash
conda create -n PsycoLLM python=3.10
conda activate PsycoLLM
pip install -r requirements.txt
```
3.运行python文件run.py
```python
deepspeed --num_gpus=2 run.py
```
4.开始交互

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).


## Citation

If this work is helpful, please kindly cite as:

```bibtex
@ARTICLE{10772313,
  author={Hu, Jinpeng and Dong, Tengteng and Luo, Gang and Ma, Hui and Zou, Peng and Sun, Xiao and Guo, Dan and Yang, Xun and Wang, Meng},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={PsycoLLM: Enhancing LLM for Psychological Understanding and Evaluation}, 
  year={2025},
  volume={12},
  number={2},
  pages={539-551},
  doi={10.1109/TCSS.2024.3497725}}
```

## Acknowledgement
感谢以下同学对本项目的帮助（排名不分先后），包括数据收集、模型评估等，赵佳琪、李绪彬、刘序雄、刘方园、李静远、戴崇远、徐开元

This repo benefits from [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl) and [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their wonderful works.
