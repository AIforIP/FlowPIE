<h3 align="center"><strong>FlowPIE</strong>: Test-Time Scientific Idea Evolution
with Flow-Guided Literature Exploration</h3>

[**🌐 Homepage**](https://flowpie.wangqiyao.me/) | [**🤗 Dataset**]() | [**📖 arXiv**](https://arxiv.org/abs/) | [**GitHub**](https://github.com/AIforIP/FlowPIE)


This repo contains the evaluation code for the paper "[FlowPIE: Test-Time Scientific Idea Evolution with Flow-Guided Literature Exploration](https://arxiv.org/abs/)"

## 🔔 News

- 🔥 [2026-01-01] Research Begining.


## 📝 Introduction 

## ✨ Highlights 

- Flow-guided MCTS: expand promising literature trajectories using flow-based scores (GFlowNet-inspired).
- GRM (Generative Reward Model): LLM-based evaluator to score ideas and guide both retrieval and evolution.
- Idea evolution at test time: selection, crossover, mutation with isolation-island parallelism to encourage cross-domain mixing and diversity.
- Provide detailed ideas, with verifiable experimental design plans.

![FlowPIE overview](./assets/overview.bmp)

## 🚀 Quick start 

Create & activate the conda environment (we use a conda env named `flowpie`):

```bash
conda create -n flowpie python=3.11 
conda activate flowpie
pip install -r requirements.txt
```

Before running the code, please make sure you have filled in all the configuration information in the config fileq.

Run Phase 1 (flow-guided MCTS). Phase1 provides a module entrypoint:

```bash
python -m src.phase1.main
```

Run Phase 2 (test-time evolution). Phase2 provides a module entrypoint:

```bash
python -m src.phase2.main
```


## Citation

When citing this work, please use the following BibTeX entry:

```bibtex

```

## Contact
`wangqiyao25@mails.ucas.ac.cn`





