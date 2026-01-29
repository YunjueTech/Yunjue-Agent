
<div align="center">
<img src="docs/assets/logo.jpeg" width="45%" alt="Yunjue Tech" />
</div>


<div align="center">

[![BLOG](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.yunjuetech.com/en)
[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YunjueTech/Yunjue-Agent)
[![Paper](https://img.shields.io/badge/Paper-De2c33?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](tech_report/YunjueAgentTechReport.pdf)

</div>

</div>

<div align="center">

### [English](README.md)｜[中文](README_zh.md)

</div>

<div align="center">

</div>

---

本仓库是 **Yunjue Agent**（云玦智能体）的官方实现。云玦科技（Yunjue Technology）是一家致力于构建**自进化 AGI（通用人工智能）和穿戴式设备的前沿科技公司。我们是一群不知疲倦的探索者，成员汇聚自顶尖 AI 实验室和工程团队。我们不满足于“静态”的大模型——即那些在训练完成瞬间参数矩阵就被冻结的模型。我们坚信，真正的智能不在于存储了多少过去的知识，而在于面对未知的未来时，具备适应、学习和创造工具**的能力。

我们欢迎各方交流。无论是融资咨询、技术探讨，还是希望加入我们的团队，请通过 qiweizhen@yunjuetech.com 联系我们。

## 📰 新闻与动态

* **[2026-01-26]** 🎉 **首次发布**：我们开源了 **Yunjue Agent** 框架！
* **[预计: 2026-01-31]** 🔜 **数据发布**：我们计划发布**五个基准数据集的完整轨迹（Traces）数据**。
* **[预计: 2026-01-31]** ✨ **纯净版发布**：我们将发布一个**清理后的易用版本**代码库，以提供更好的可访问性和集成体验。

> **⚠️ 关于当前版本的说明**：当前代码库是基于我们研究实验重构的初始版本。虽然我们已经验证了核心逻辑，但在复现过程中可能会遇到少量 Bug 或边缘情况。我们正在持续清理代码，欢迎提交 Issue 或 PR！

---

## 🚀 快速开始 (Quick Start)

### 📋 环境要求

* **Python**: 3.12 或更高版本
* **包管理器**: [`uv`](https://www.google.com/search?q=%5Bhttps://docs.astral.sh/uv/%5D(https://docs.astral.sh/uv/))
* **操作系统**: Linux

### ⚡ 快速安装

**示例**：从 DeepSearchQA 数据集开始进化。

```bash
# 1. 克隆并设置
git clone https://github.com/YunjueTech/Yunjue-Agent.git && cd Yunjue-Agent

chmod +x install.sh

./install.sh

cp .env.example .env

cp conf.yaml.example conf.yaml

source .venv/bin/activate

./scripts/evolve.sh --dataset DEEPSEARCHQA --run_name test --batch_size 1 --start 0

```

🎉 **预期输出**：你的 Agent 将开始完成 DeepSearchQA 中的问题。你可以在 `output/test` 中查看相应的日志 😊

---

## 🤖 什么是 Yunjue Agent?

传统的 Agent 系统通常在开放环境中表现挣扎，因为任务分布不断变化且缺乏外部监督。它们依赖静态工具集或离线训练，滞后于环境动态，导致系统的能力边界僵化且未知。为了解决这个问题，我们提出了 **原位自进化（In-Situ Self-Evolving）** 范式。

这种方法将连续的任务交互视为持续的经验流，使系统能够在没有真实标签的情况下，将短期执行反馈提炼为长期的、可复用的能力。在此框架内，我们将 **工具进化** 视为能力扩展的关键路径，因为它提供了可验证的二元反馈信号。基于此框架，我们开发了 **Yunjue Agent**，这是一个能够迭代合成、优化和复用工具以应对新挑战的系统。

为了优化进化效率，我们进一步引入了 **并行批量进化** 策略。在零起点设置下对五个不同基准进行的实证评估表明，该系统显著优于专有基线模型。此外，补充的热启动评估证实，积累的通用知识可以无缝迁移到新领域。最后，我们提出了一种监控进化收敛的新指标，其功能类似于传统优化中的训练损失。我们开源了代码库、系统轨迹和进化后的工具，以促进对弹性、自进化智能的未来研究。

<div align="center">
<img src="docs/assets/overview.png" width="100%" alt="Yunjue Agent 架构概览">
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

## 🌟 核心亮点

* **🧬 原位自进化范式**
我们引入了一种新颖的 Agent 学习框架，弥合了静态能力与即时进化之间的鸿沟。通过将离散交互重构为连续的经验流，系统通过内部反馈循环将短期推理提炼为长期能力。这使得 Agent 无需额外的监督信号，即可在开放环境中实现实时适应和探索。
* **🚀 "白板"起步即达 SOTA**
从**空工具库**开始，我们的系统仅依靠推理时的生成、验证和归纳，就达到了最先进的性能。它表现出了相对于后端模型的显著提升（例如，在 DeepSearchQA 上比 Gemini 3 Pro 提升了 **+17.4%**），并在 **HLE 排行榜上获得第 2 名**，证明了从零开始自举通用能力的可行性。
* **🛠️ "工具优先"的进化原则**
我们将工具进化置于记忆或工作流之上，作为能力的主要驱动力。工具提供客观的 **二元反馈**，在缺乏人类标注的情况下充当可靠的内部监督信号。这种方法降低了幻觉风险并防止了策略偏差，确保了通用原语的稳定积累。
* **🔍 完全可复现与开放轨迹**
我们发布了一套全面的开放资产，包括端到端代码、基准测试脚本、版本化的工具制品和完整的交互轨迹。这将“黑盒”Agent 结果转化为透明、可审计的研究，使研究人员能够对工具收敛性、进化效率和合并策略进行细粒度分析。

## 📈 基准测试表现

我们在 **HLE**、**DeepSearchQA**、**FinSearchComp (T2&T3)**、**xbench-ScienceQA** 和 **xbench-DeepSearch** 等一系列基准测试中对 Yunjue Agent 进行了评测，并取得了 SOTA 结果。

<img width="100%" alt="主要结果图表" src="docs/assets/main_results.jpeg" />

---

## 📄 许可证

本项目采用 Apache License 2.0 许可证。

---
