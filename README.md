# 
<div align="center">

# FreePRM: Training Process Reward Models Without Ground Truth Process Labels

</div>

<p align="center">
  📄 <a href="https://openreview.net/pdf?id=VsqQzsMYbg" target="_blank">Paper</a> &nbsp; | &nbsp;
  🌐 <a href="https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection" target="_blank">Dataset</a> &nbsp; | &nbsp;
  📘 <a href="https://huggingface.co/GAIR/ToRL-7B" target="_blank">Model</a>
</p>


<div align="center">
<img src="assets/model_scheme.png" width="700" alt="torl-abstarct-1">
</div>

>  The input consists of question and its solution. The solution is divided into multiple steps, each separated by a special token: (“\n\n\n\n\n”). Each step is pseudo-labeled as either "+" (correct) or "-" (incorrect). FreePRM predicts three probabilities for each step: wrong, buffer, and right. When a step is labeled as incorrect ("-"), the training target is set to [1, 0], meaning the combined probability of wrong and buffer should sum to 1, while the probability of right is expected to be 0.


<div align="center">
<img src="assets/bon.png" width="700" alt="torl-abstarct-2">
</div>

> BoN results with different generation models on MATH-500. we generate 128 solutions for
each problem, and for each solution, we pich the final step reward as the overall score. It shows that FreePRM matches or exceeds PRM which is trained on labeled process data, consistently outperforms Majority Voting and ORM across diverse model scales.


## Releases

[2025/05/30] We're releasing the following components:

- 🚀 **Training**: Complete implementation of our training pipeline
- 🔥 **[FreePRM Dataset](https://github.com/GAIR-NLP/ToRL/tree/main/data/torl_data)**: Our curated dataset of 28k mathematical questions
- 🤖 **[FreePRM Model](https://huggingface.co/GAIR/ToRL)**: Model training with FreePRM.

## Overview

Recent advancements in Large Language Models (LLMs) have demonstrated that Process Reward Models (PRMs) play a crucial role in enhancing model performance. However, training PRMs typically requires step-level labels, either manually annotated or automatically generated, which can be costly and difficult to obtain at scale. To address this challenge, we introduce FreePRM, a weakly supervised framework for training PRMs without access to ground-truth step-level labels. FreePRM first generates pseudo step-level labels based on the correctness of final outcome, and then employs Buffer Probability to eliminate impact of noise inherent in pseudo labeling. Experimental results show that FreePRM achieves an average F1 score of 53.0% on ProcessBench, outperforming fully supervised PRM trained on Math-Shepherd by +24.1%. Compared to other open-source PRMs, FreePRM outperforms upon RLHFlow-PRM-Mistral-8B (28.4%) by +24.6%, EurusPRM (31.3%) by +21.7%, and Skywork-PRM-7B (42.1%) by +10.9%. This work introduces a new paradigm in PRM training, significantly reducing reliance on costly step-level annotations while maintaining strong performance. 

### FreePRM Performance


Here's the converted Markdown version of your LaTeX table content:

---

### ProcessBench Results

FreePRM outperforms baseline methods. Although there remains a performance gap compared to *Qwen2.5-Math-7B-PRM800K* (trained on high-cost manually labeled data), this difference is relatively small, especially considering our model is trained entirely without process labels.

| Model                            | Process Label | GSM8K | MATH | Olympiad Bench | Omni-MATH | Avg. |
|----------------------------------|---------------|-------|------|----------------|-----------|------|
| Qwen2.5-Math-7B-PRM800K          | Y (manual label) | **68.2** | **62.6** | **50.7**     | **44.3** | **56.5** |
| Math-Shepherd-PRM-7B             | Y             | 47.9  | 29.5 | 24.8           | 23.8      | 31.5 |
| RLHFlow-PRM-Mistral-8B           | Y             | 50.4  | 33.4 | 13.8           | 15.8      | 28.4 |
| RLHFlow-PRM-Deepseek-8B          | Y             | 38.8  | 33.8 | 16.9           | 16.9      | 26.6 |
| Skywork-PRM-7B                   | Y             | 70.8  | 53.6 | 22.9           | 21.0      | 42.1 |
| EurusPRM-Stage1                  | Y             | 44.3  | 35.6 | 21.7           | 23.1      | 31.2 |
| EurusPRM-Stage2                  | Y             | 47.3  | 35.7 | 21.2           | 20.9      | 31.3 |
| Qwen2.5-Math-7B-Math-Shepherd-PRM| Y             | 62.5  | 31.6 | 13.7           | 7.7       | 28.9 |
| **FreePRM-7B-Math-Shepherd (ours)** | N          | **74.2** | _58.8_ | _39.0_       | _40.1_   | _53.0_ |




## Quick Start

### Environment setup
```
pip install -r requirements.txt
```

### Training

Execute the following command to train the model.
```
CUDA_VISIBLE_DEVICES=0,1 python train_prm.py
```

### Evaluation

Execute the following command to evaluate the trained model.
```
CUDA_VISIBLE_DEVICES=0 python server_prm.py

python evaluate_prm.py
```

## Acknowledgements

We appreciate the work of [ProcessBench](https://arxiv.org/pdf/2412.06559), which provides a valuable evaluation baseline for model performance. We also thank the [Qwen-Math](https://github.com/QwenLM/Qwen2.5-Math) team for open-sourcing their models, and acknowledge [TRL](https://github.com/huggingface/trl) and [vLLM](https://github.com/vllm-project/vllm) for providing the essential training framework and inference infrastructure, respectively, that enabled this research.


## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{ESbK4EbZzw,
      title={FreePRM: Training Process Reward Models Without Ground Truth Process Labels}, 
      author={Lin Sun, Chuang Liu, Xiaofeng Ma, Tao Yang, Weijia Lu, Ning Wu},
      year={2025},
      primaryClass={cs.CL},
      url={https://openreview.net/pdf?id=ESbK4EbZzw}, 
}
```
