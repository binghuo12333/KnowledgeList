# 👋 欢迎来到我的个人知识库

本仓库是个人知识沉淀与实战总结中心，**以 AI 大模型技术为核心**，整合产品经理方法论、项目管理实战工具及数据应用技巧，所有内容均附实战链接与资源对比，供学习交流使用。

## 📚 知识体系总览

|领域分类|核心内容|资源状态|实战链接|
|---|---|---|---|
|**AI 大模型（重点）**|开源大模型、图像模型、Embedding/Reranker、智能体、协议调用|✅ 深度解析+实战案例|见下方详细板块|
|产品经理|书籍笔记、产品设计、文档模板、工具应用|✅ 分类整理+工具链接|见下方详细板块|
|项目管理|数据模板、协作方法、实战案例|✅ 可直接复用资源|见下方详细板块|
|工具集|数据库、数据分析、提示词调优工具|✅ 工具对比+使用指南|见下方详细板块|
## 🤖 核心板块：AI 大模型技术栈

**聚焦实战应用与资源对比**，涵盖主流模型解析、协议调用、智能体开发及插件集成，所有分析均基于实测数据，保证对比准确性。

### 1. 开源大模型系列（实测对比）

针对主流开源大模型系列，从性能、资源需求、适用场景三维度实测分析，附部署教程与调优方案。

|模型系列|核心优势|资源需求（CPU/GPU）|适用场景|实测链接|
|---|---|---|---|---|
|Llama 系列（Meta）|通用能力强、生态完善、微调门槛低|7B：16G GPU；70B：80G+ GPU|通用对话、内容生成、企业级应用|[实测报告+部署教程](https://github.com/你的用户名/你的仓库名/tree/main/AI/大模型/开源模型/Llama系列)|
|Qwen 系列（阿里）|中文能力优异、多模态支持、轻量化版本丰富|7B：12G GPU；32B：48G GPU|中文内容创作、本地化部署、多模态交互|[实测报告+部署教程](https://github.com/你的用户名/你的仓库名/tree/main/AI/大模型/开源模型/Qwen系列)|
|Mistral 系列|推理速度快、上下文窗口大、效率优先|7B：14G GPU；13B：24G GPU|实时对话、高并发场景、边缘设备部署|[实测报告+部署教程](https://github.com/你的用户名/你的仓库名/tree/main/AI/大模型/开源模型/Mistral系列)|
|Falcon 系列（IBM）|商业友好许可、大参数量版本性能强劲|40B：64G GPU；180B：128G+ GPU|企业级解决方案、大规模文本处理|[实测报告+部署教程](https://github.com/你的用户名/你的仓库名/tree/main/AI/大模型/开源模型/Falcon系列)|
### 2. 专项模型解析

- **图像模型**：涵盖 Stable Diffusion、MidJourney（API 调用）、DALL·E 系列，对比生成质量、速度、自定义能力，附 Prompt 工程技巧与微调案例。[详细解析](https://github.com/你的用户名/你的仓库名/tree/main/AI/大模型/专项模型/图像模型)

- **Embedding & Reranker 模型**：实测 BGE、m3e、E5 等 Embedding 模型语义相似度能力，对比 Cohere Rerank、Cross-BERT 等 Reranker 效果，附检索系统搭建实战。[详细解析](https://github.com/你的用户名/你的仓库名/tree/main/AI/大模型/专项模型/Embedding-Reranker)

### 3. 智能体平台开发

基于开源大模型构建自定义智能体，整合工具调用、多轮对话逻辑，附平台搭建源码与应用案例。

- 主流智能体框架对比：LangChain、AutoGPT、AgentGPT 实战适配

- 自定义智能体开发教程：任务拆解、工具集成、异常处理

- 实战案例：文档问答智能体、多模态交互智能体

- 链接：[智能体开发全流程](https://github.com/你的用户名/你的仓库名/tree/main/AI/智能体平台)

### 4. MPC 插件与大模型协议调用

详解大模型相关协议使用方法，包括 MPC（多方安全计算）插件集成、API 调用规范、本地部署协议适配。

- 核心协议：OpenAI API、Anthropic API、本地化模型 OpenLLM 协议

- MPC 插件：隐私计算场景下的大模型插件集成、数据加密传输方案

- 调用实战：Python/Java 调用示例、批量请求优化、错误重试机制

- 链接：[协议文档+调用源码](https://github.com/你的用户名/你的仓库名/tree/main/AI/协议与插件)

## 🛠️ 工具集：数据库、分析与调优

### 1. 数据库与存储工具

|工具类型|推荐工具|核心用途|使用链接|
|---|---|---|---|
|向量数据库|Milvus、Pinecone、Chroma|存储 Embedding 向量、支持高效检索|[使用指南](https://github.com/你的用户名/你的仓库名/tree/main/工具集/数据库/向量数据库)|
|关系型数据库|MySQL、PostgreSQL|结构化数据存储、大模型训练数据管理|[优化技巧](https://github.com/你的用户名/你的仓库名/tree/main/工具集/数据库/关系型数据库)|
|文档数据库|MongoDB、CouchDB|非结构化数据存储、智能体记忆管理|[实战案例](https://github.com/你的用户名/你的仓库名/tree/main/工具集/数据库/文档数据库)|
### 2. 数据分析工具

- **核心工具**：Python（Pandas、NumPy）、SQL、Tableau、Power BI

- **实战场景**：大模型性能数据分析、用户行为分析、项目数据可视化

- **链接**：[脚本模板+可视化案例](https://github.com/你的用户名/你的仓库名/tree/main/工具集/数据分析)

### 3. 提示词调优工具

专注提升大模型输出质量，整合调优方法论与工具，适配不同模型特性。

- 调优工具：PromptBase、LangSmith、ChatGPT Prompt Tuner

- 核心方法：Few-Shot、Chain-of-Thought、Role Prompting 实战

- 链接：[调优指南+案例库](https://github.com/你的用户名/你的仓库名/tree/main/工具集/提示词调优)

## 🎯 产品经理知识与工具

### 1. 核心知识点（附链接）

- 产品设计：用户画像、需求拆解、原型设计、交互逻辑 [详细笔记](https://github.com/你的用户名/你的仓库名/tree/main/产品经理/产品设计)

- 需求管理：PRD 撰写、需求优先级排序、用户故事设计 [模板+案例](https://github.com/你的用户名/你的仓库名/tree/main/产品经理/需求管理)

- 产品迭代：MVP 设计、迭代规划、数据驱动优化 [实战方法](https://github.com/你的用户名/你的仓库名/tree/main/产品经理/产品迭代)

- AI 产品：大模型产品落地、Prompt 工程在产品中的应用 [专项解析](https://github.com/你的用户名/你的仓库名/tree/main/产品经理/AI产品)

### 2. 必备工具（附官方链接）

|工具类别|工具名称|核心用途|官方链接|
|---|---|---|---|
|原型设计|Figma、Axure RP、墨刀|交互原型、视觉设计、团队协作|[Figma](https://www.figma.com/) / [Axure](https://www.axure.com/)|
|文档编辑|Notion、飞书文档、Confluence|PRD 撰写、知识库管理、团队协同|[Notion](https://www.notion.so/) / [飞书文档](https://www.feishu.cn/)|
|用户调研|问卷星、麦客表单、UserTesting|用户需求收集、行为分析、反馈整理|[问卷星](https://www.wjx.cn/) / [UserTesting](https://www.usertesting.com/)|
## 📋 项目管理实战

### 1. 核心资源（附链接）

- 数据模板：甘特图、风险评估矩阵、资源分配表、项目复盘模板 [可直接复用](https://github.com/你的用户名/你的仓库名/tree/main/项目管理/数据模板)

- 协作方法：敏捷 Scrum 实战、跨部门沟通技巧、项目进度追踪[笔记总结](https://github.com/你的用户名/你的仓库名/tree/main/项目管理/协作方法)

- AI 项目管理：大模型开发项目排期、资源预估、风险管控 [专项指南](https://github.com/你的用户名/你的仓库名/tree/main/项目管理/AI项目专项)

### 2. 工具链接

- 项目协作：Jira [官方链接](https://www.atlassian.com/software/jira)、飞书项目 [使用指南](https://www.feishu.cn/hc/zh-CN/articles/360024984773)

- 进度管理：Trello、Microsoft Project

- 团队沟通：飞书、Slack

## 📈 近期更新计划

1. 新增 Llama 4 与 Qwen 2.0 实测对比，补充最新开源模型数据

2. 完善 RAG 系统搭建全流程（结合 Embedding+向量数据库+大模型）

3. 新增产品经理 AI 工具实战案例（Prompt 驱动的需求分析）

4. 补充 MPC 插件在隐私计算场景的完整部署教程


## 📞 联系我
- GitHub: [binghuo12333](https://github.com/binghuo12333)
- 邮箱: binghuo12333@gmail.com

<div align="center">
  <sub>最后更新于 2026 年 1 月</sub>
</div>
