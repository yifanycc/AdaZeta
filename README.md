# Source Code for paper 'AdaZeta: Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient Large Language Models Fine-Tuning'
2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)

Yifan Yang (UCSB), Kai Zhen (Amazon AGI), Ershad Banijamali (Amazon AGI), Athanasios Mouchtaris (Amazon AGI), Zheng Zhang (UCSB)

---

This is the implementation for the paper [AdaZeta: Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient Large Language Models Fine-Tuning](https://arxiv.org/pdf/2406.18060).  In this paper, we propose the Adaptive Zeroth-order Tensor-Train Adaption (AdaZeta) framework, specifically designed to improve the performance and convergence of the ZO methods. To enhance
dimension-dependent ZO estimation accuracy, we introduce a fast-forward, low-parameter tensorized adapter. To tackle the frequently observed divergence issue in large-scale ZO finetuning tasks, we propose an adaptive query number schedule that guarantees convergence. Detailed theoretical analysis and extensive experimental results on Roberta-Large and Llama-2-7B models substantiate the efficacy of our AdaZeta framework in terms of accuracy, memory efficiency, and convergence speed.

<h1> <p>🤗 News</p></h1>


**09/20/2024: ~~We plan to release the formal source code with detailed documentation around the first week of Oct, stay tuned.~~ (So sad, I'm entangled by the NAACL due... I will work on the doc, but a bit behind schedule)** 

**09/20/2024:** Our paper 'AdaZeta: Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient Large Language Models Fine-Tuning'
has been accepted by the EMNLP 2024 main conference
