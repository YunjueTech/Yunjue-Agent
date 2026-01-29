<div align="center">
  <img src="docs/assets/logo.jpeg" width="45%" alt="Yunjue Tech" />
</div>

<br> 


<div align="center">

[![BLOG](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.yunjuetech.com/en)
[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YunjueTech/Yunjue-Agent)
[![Paper](https://img.shields.io/badge/Paper-De2c33?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](tech_report/YunjueAgentTechReport.pdf)

</div>

<div align="center">

### [English](README.md)｜[中文](README_zh.md)

</div>


---

This repo is an official implementation of Yunjue Agent by Yunjue Technology. Our company is a cutting-edge technology company dedicated to building Self-Evolving AGI (Artificial General Intelligence) and wearable devices. We are a group of tireless explorers, with members from top AI laboratories and engineering teams. We are not satisfied with "static" large models—those parameter matrices frozen at the moment of training completion. We believe that true intelligence lies not in how much past knowledge is stored, but in the ability to adapt, learn, and create tools when facing an unknown future. 

We welcome connections of all kinds. For financing inquiries, technical exchanges, or to join our team, please contact qiweizhen@yunjuetech.com

## 📰 News & Updates

- **[2026-01-26]** 🎉 **Initial Release**: We have open-sourced the **Yunjue Agent** framework!
- **[Expected: 2026-01-31]** 🔜 **Data Release**: We are scheduling the release of **full traces for the five benchmark datasets.**
- **[Expected: 2026-01-31]** ✨ **Clean Version Release**: We will release a **cleaned, easy-to-use version** of the codebase for better accessibility and integration.

> **⚠️ Note on Current Release**: The current codebase is an initial release refactored from our research experiments. While we have verified the core logic, there might be minor bugs or edge cases during reproduction. We are continuously cleaning up the code and welcome any issues or PRs!
---

## 🚀 Quick Start

### 📋 Prerequisites

- **Python**: 3.12 or higher
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Operating System**: Linux

### ⚡ Quick Setup

**Example**: Start evolving from DeepSearchQA.

```bash
# 1. Clone and setup
git clone https://github.com/YunjueTech/Yunjue-Agent.git && cd Yunjue-Agent

chmod +x install.sh

./install.sh

cp .env.example .env

cp conf.yaml.example conf.yaml

source .venv/bin/activate

./scripts/evolve.sh --dataset DEEPSEARCHQA --run_name test --batch_size 1 --start 0

```

🎉 **Expected Output:** Your agent will start completing questions from DeepSearchQA. You can view the corresponding logs in `output/test` 😊

---

## 🤖 What is Yunjue Agent?

Conventional agent systems often struggle in open-ended environments where task distributions continuously drift and external supervision is scarce. Their reliance on static toolsets or offline training lags behind these dynamics, leaving the system's capability boundaries rigid and unknown. To address this, we propose the *In-Situ Self-Evolving* paradigm. This approach treats sequential task interactions as a continuous stream of experience, enabling the system to distill short-term execution feedback into long-term, reusable capabilities without access to ground-truth labels. Within this framework, we identify *tool evolution* as the critical pathway for capability expansion, which provides verifiable, binary feedback signals. Within this framework, we develop *Yunjue Agent*, a system that iteratively synthesizes, optimizes, and reuses tools to navigate emerging challenges. To optimize evolutionary efficiency, we further introduce a *Parallel Batch Evolution* strategy. Empirical evaluations across five diverse benchmarks under a zero-start setting demonstrate significant performance gains over proprietary baselines. Additionally, complementary warm-start evaluations confirm that the accumulated general knowledge can be seamlessly transferred to novel domains. Finally, we propose a novel metric to monitor evolution convergence, serving as a function analogous to training loss in conventional optimization. We open-source our codebase, system traces, and evolved tools to facilitate future research in resilient, self-evolving intelligence.

<div align="center">
<img src="docs/assets/overview.png" width="100%" alt="Yunjue Agent Architecture">
</div>

<table align="center" style="border: 1px solid #ccc; border-radius: 8px; padding: 12px; background-color: #f9f9f9; width: 60%;">
  <tr>
    <td style="text-align: center; padding: 10px;">
      <strong>Demo</strong> 
      <br>
      <video src="https://github.com/user-attachments/assets/6bcc6aa2-f982-4685-8b2a-846e4a03bf3e"
             controls muted preload="metadata"
             width="50%" height="50%"
      </video>
    </td>
  </tr>
</table>

---

## 🌟 Highlights

- **🧬 In-situ Self-evolving Paradigm**
    
    We introduce a novel agentic learning framework that bridges the gap between static capability and on-the-fly evolving. By reframing discrete interactions as a continuous stream of experience, the system distills short-term inference into long-term capabilities via internal feedback loops. This enables real-time adaptation and exploration in open-ended environments without the need for additional supervision signals.
    
- **🚀 SOTA Performance from "Tabula Rasa"**
    
    Starting with an **empty tool library** (Zero-Start), our system achieves State-of-the-Art performance by relying solely on inference-time generation, verification, and induction. It demonstrates significant gains over backend models (e.g., **+17.4%** on DeepSearchQA over Gemini 3 Pro) and secures **2nd place on the HLE leaderboard**, proving the feasibility of bootstrapping general capabilities from scratch.
    
- **🛠️ "Tool-First" Evolutionary Principle**
    
    We prioritize tool evolution over Memory or Workflows as the primary driver of capability. Tools provide objective **Binary Feedback** (via code execution success/failure), serving as a reliable internal supervision signal in the absence of human annotation. This approach mitigates hallucination risks and prevents strategy bias, ensuring stable accumulation of general primitives.
    
- **🔍 Fully Reproducible & Open Traces**
    
    We release a comprehensive open-asset suite, including end-to-end code, benchmark scripts, versioned tool artifacts, and full interaction traces. This transforms "black-box" agent results into transparent, auditable research, enabling granular analysis of tool convergence, evolution efficiency, and merging strategies.

## 📈 Performance on Benchmarks

We benchmark Yunjue Agent on a series of benchmarks, including **HLE**, **DeepSearchQA**, **FinSearchComp (T2&T3)**, **xbench-ScienceQA** and **xbench-DeepSearch**, and achieved SOTA results.

<img width="100%" alt="image" src="docs/assets/main_results.jpeg" />



---


## 📄 License

This project is licensed under the Apache License 2.0.
