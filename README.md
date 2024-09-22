[//]: # (![# PsycoLLM]&#40;assets/logo.png&#41;)

# PsycoLLM: Enhancing LLM for Psychological Understanding and Evaluation


## Introduction

Here is an overview of dataset preparation:
![Overview of dataset preparation.](assets/dataset_overview.jpg)

A comprehensive pipeline is used to generate multi-turn dialogue, which includes stages of generation, evidence support, and refinement:
![Examples of the generated multi-turn dialogue data.](assets/multi_turn_example_v2.jpg)

The process for knowledge-based QA generation is:
![Knowledge-based QA generation.](assets/knowledge-base-QA.jpg)

## Highlights

- We develop a benchmark based on official psychological examinations in China.

- PsycoLLM achieve an accuracy rate surpassing 60\% in averaged MCQs in the proposed benchmark.


## License

This repository is licensed under the [Apache-2.0 License](LICENSE).


## Citation

If this work is helpful, please kindly cite as:

```bibtex
@misc{hu2024psycollmenhancingllmpsychological,
      title={PsycoLLM: Enhancing LLM for Psychological Understanding and Evaluation}, 
      author={Jinpeng Hu and Tengteng Dong and Hui Ma and Peng Zou and Xiao Sun and Meng Wang},
      year={2024},
      eprint={2407.05721},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.05721}, 
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl) and [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their wonderful works.




Paper here 👇
<img src="https://img.shields.io/badge/python-3.8+-blue.svg" /><a href='https://arxiv.org/pdf/2407.05721'><img src='https://img.shields.io/badge/ArXiv-2407.05721v2-red'></a>

## 最近更新 Latest News

-🥰 [2024.9.22] 我们的PsycoLLM大模型正式发布了！模型地址：[MACLAB-HFUT/PsycoLLM](https://huggingface.co/MACLAB-HFUT/PsycoLLM)

## 项目简介 Project introduction

-👀 **背景**：2022年4月27日，国务院办公厅正式印发《“十四五”国民健康规划》，首次将心理健康纳入国家发展目标，标志着我国对心理健康战略地位的重大提升。此前，我国已陆续出台《中华人民共和国精神卫生法》、《国家心理健康工作规划（2019-2025）》以及《国家心理健康教育与推广纲要（2021-2025）》等一系列法律法规，彰显了国家对心理健康的日益重视。据 《2023年度中国精神心理健康蓝皮书》（下称蓝皮书）数据显示，我国成人抑郁风险检出率为10.6%，焦虑风险检出率为15.8%。仅有36%的国民认为自己的心理健康状况良好，而在自评为“较差”的群体中，抑郁风险检出率更是高达45.1%。
-🤔 **简介**：基于此，我们基于Qwen
<p align="center">
    <img src="./assets/multi_turn_example_v2.png" width=900px/>
