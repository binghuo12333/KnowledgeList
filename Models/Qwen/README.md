# Qwen（通义千问）系列模型使用指南

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i4/O1CN01nqE6sX1zGbH9M6U6O_!!6000000006571-2-tps-200-200.png" width="120" alt="Qwen Logo">
  <p>阿里云通义千问全系列最新开源模型部署与使用指南</p>
  <a href="https://github.com/QwenLM/Qwen"><img src="https://img.shields.io/github/stars/QwenLM/Qwen?style=social"></a>
  <a href="https://modelscope.cn/organization/qwen"><img src="https://img.shields.io/badge/ModelScope-Qwen-ff69b4"></a>
  <a href="https://huggingface.co/collections/Qwen"><img src="https://img.shields.io/badge/HuggingFace-Qwen-yellow"></a>
</div>

## 🔗 官方资源

| 平台 | 链接 | 核心用途 |
|------|------|----------|
| 阿里云百炼控制台 | [https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/all](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/all) | 在线部署、API调用、调试 |
| Qwen 官方仓库 | [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen) | 源码、部署脚本、示例 |
| Hugging Face 模型库 | [https://huggingface.co/collections/Qwen](https://huggingface.co/collections/Qwen) | 模型下载、HF生态适配 |
| 魔搭社区 | [https://modelscope.cn/organization/qwen](https://modelscope.cn/organization/qwen) | 中文生态、一键部署 |

## 📚 模型分类及使用说明

### 1. 通用大语言模型（Qwen-2 Chat）
> 最新迭代版本，推理速度提升30%、显存占用降低15%，支持超长上下文

#### 📋 最新模型列表
- Qwen-2-0.5B-Chat（超轻量版，纯CPU运行）
- Qwen-2-1.5B-Chat（轻量版，低显存显卡适配）
- Qwen-2-7B-Chat（主流版，消费级显卡首选）
- Qwen-2-14B-Chat（平衡版，性能与资源兼顾）
- Qwen-2-72B-Chat（旗舰版，128K超长上下文）
- Qwen-2-110B-Chat（超大版，企业级部署）

#### 🚀 快速启动
```bash
# 安装核心依赖
pip install "fschat[model_worker,webui]" transformers torch accelerate sentencepiece protobuf

# 1. 命令行交互（INT4量化，低显存）
python -m fastchat.serve.cli \
  --model-path ./Qwen-2-7B-Chat \
  --load-4bit \
  --temperature 0.7 \
  --trust-remote-code

# 2. WebUI可视化服务（访问：http://localhost:7860）
python -m fastchat.serve.controller &
python -m fastchat.serve.model_worker --model-path ./Qwen-2-7B-Chat --device cuda --load-4bit &
python -m fastchat.serve.gradio_web_server

# 3. OpenAI兼容API服务（端口8000）
python -m fastchat.serve.openai_api_server \
  --model-path ./Qwen-2-7B-Chat \
  --host 0.0.0.0 \
  --port 8000 \
  --load-4bit

### 2. 多模态模型（Qwen-VL/Audio 2.0）
<div align="left">
  <img src="https://img.shields.io/badge/Multimodal-VL/Audio%202.0-9cf" alt="Multimodal">
  <img src="https://img.shields.io/badge/Context-8K-important" alt="Context">
</div>

> **核心特性**
> - 🖼️ Qwen-VL 2.0：支持4K分辨率图片、多图对比、复杂图表/OCR分析
> - 🎙️ Qwen-Audio 2.0：多语言语音识别/合成、语音翻译、音频理解
> - �融合版：图文音多模态交互，跨模态语义理解能力行业领先

#### 📋 最新模型列表
| 模型名称 | 适用场景 | 核心优势 |
| :------- | :------- | :------- |
| `Qwen-VL-2-7B-Chat` | 通用图文交互 | 轻量高效，消费级显卡可运行 |
| `Qwen-VL-2-14B-Chat` | 复杂图文分析 | 高精度图表解读、多图推理 |
| `Qwen-Audio-2-7B-Chat` | 语音交互 | 低延迟语音识别，支持10+语言 |
| `Qwen-VL-Audio-2-7B-Chat` | 全模态交互 | 图文音一体化理解与生成 |

#### 🚀 快速启动
##### 环境准备
```bash
# 安装多模态专属依赖
pip install torchvision pillow soundfile librosa transformers accelerate opencv-python pydub

1. 命令行图文对话
# Qwen-VL-2-7B-Chat（8bit量化）
python -m fastchat.serve.cli \
  --model-path ./Qwen-VL-2-7B-Chat \
  --load-8bit \
  --trust-remote-code \
  --image ./demo_images/chart.png  # 替换为本地图片路径

2. WebUI 可视化服务（支持图片 / 音频上传）
# 后台启动（推荐生产环境）
nohup python -m fastchat.serve.controller > controller.log 2>&1 &
nohup python -m fastchat.serve.model_worker \
  --model-path ./Qwen-VL-Audio-2-7B-Chat \
  --device cuda \
  --load-4bit > model_worker.log 2>&1 &
nohup python -m fastchat.serve.gradio_web_server \
  --multimodal \
  --server-port 7860 > webui.log 2>&1 &

# 访问地址：http://localhost:7860

3. 嵌入模型（Qwen-Embedding V2）
<div align="left"><img src="https://img.shields.io/badge/Embedding-V2-9cf" alt="Embedding"><img src="https://img.shields.io/badge/Dimension-768/1024-important" alt="Dimension"><img src="https://img.shields.io/badge/Context-8K-success" alt="Context"></div>
核心特性
🌐 中英文双语嵌入，语义对齐效果优于主流开源模型
📜 长文本分段嵌入，支持 8K 文本长度
⚡ 推理速度提升 50%，适配高并发检索场景
🎯 数学 / 代码专用版，垂直领域效果优化

🚀 快速使用
基础嵌入生成
python运行
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Embedding-V2 文本嵌入示例"""
from transformers import AutoModel, AutoTokenizer
import torch

# 环境配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 加载模型（建议提前下载到本地）
tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen-Embedding-V2",
    trust_remote_code=True,
    cache_dir="./cache"
)
model = AutoModel.from_pretrained(
    "./Qwen-Embedding-V2",
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE,
    cache_dir="./cache"
).to(DEVICE).eval()

# 文本嵌入生成
def get_text_embedding(texts: list) -> torch.Tensor:
    """
    生成文本嵌入向量
    :param texts: 文本列表
    :return: 归一化后的嵌入向量 [batch_size, dim]
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 取<bos> token作为句子嵌入
        embeddings = outputs.last_hidden_state[:, 0]
        # 向量归一化（检索场景必做）
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings

# 示例调用
if __name__ == "__main__":
    test_texts = [
        "通义千问是阿里云推出的开源大语言模型",
        "Qwen-2 is the latest open-source LLM by Alibaba Cloud",
        "Qwen-Embedding-V2 支持长文本语义向量生成"
    ]
    
    embeddings = get_text_embedding(test_texts)
    print(f"嵌入向量维度：{embeddings.shape}")  # torch.Size([3, 1024])
    print(f"第一条文本向量前5维：{embeddings[0][:5].cpu().numpy()}")
    
    # 计算相似度
    sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    print(f"中英文语义相似度：{sim.item():.4f}")

批量嵌入生成（生产级）python运行
# 批量处理（避免OOM）
def batch_get_embeddings(texts: list, batch_size: int = 32) -> list:
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = get_text_embedding(batch_texts)
        embeddings_list.append(batch_embeddings.cpu())
    return torch.cat(embeddings_list, dim=0)

# 使用示例
# large_text_list = ["文本1", "文本2", ..., "文本10000"]
# all_embeddings = batch_get_embeddings(large_text_list, batch_size=32)

4. 代码模型（Qwen-Coder 2）
<div align="left"><img src="https://img.shields.io/badge/Coder-2.x-9cf" alt="Coder"><img src="https://img.shields.io/badge/Languages-20%2B-important" alt="Languages"><img src="https://img.shields.io/badge/Context-8K-success" alt="Context"></div>
核心特性
💻 支持 Python/Java/C++/Go/JavaScript 等 20 + 编程语言
🔧 代码生成 / 补全 / 调试 / 重构 / 单元测试生成
📝 代码解释 / 性能优化 / 错误修复
🎯 编程题解答，支持 ACM/OJ 格式

🚀 快速启动
1. 命令行代码对话——bash运行
# Qwen-Coder-2-7B-Chat（4bit量化）
python -m fastchat.serve.cli \
  --model-path ./Qwen-Coder-2-7B-Chat \
  --load-4bit \
  --trust-remote-code \
  --prompt-template qwen_coder \
  --temperature 0.2 \
  --max-new-tokens 2048
2. API 服务部署（代码补全场景）——bash运行
# 启动OpenAI兼容API
python -m fastchat.serve.openai_api_server \
  --model-path ./Qwen-Coder-2-7B-Chat \
  --host 0.0.0.0 \
  --port 8000 \
  --load-4bit \
  --trust-remote-code

# 代码补全调用示例（curl）
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen-Coder-2-7B-Chat",
    "prompt": "def quick_sort(arr):\n    # 快速排序实现",
    "max_tokens": 512,
    "temperature": 0.1,
    "stop": ["\ndef", "\nclass"]
  }'
3. 代码调试专用启动——bash运行
python -m fastchat.serve.cli \
  --model-path ./Qwen-Coder-Debug-7B \
  --load-4bit \
  --trust-remote-code \
  --temperature 0.0

5. 重排序模型（Qwen-Rerank）
<div align="left"><img src="https://img.shields.io/badge/Rerank-M3-9cf" alt="Rerank"><img src="https://img.shields.io/badge/Latency-<10ms-important" alt="Latency"><img src="https://img.shields.io/badge/Context-512-success" alt="Context"></div>
核心特性
🔍 检索问答（RAG）场景专用，提升检索准确率 30%+
📚 多粒度文本重排，支持短句 / 长文本 / 跨语言重排
⚡ 轻量级模型，单条推理耗时 < 10ms
🎯 适配 ES/FAISS/PGVector 等检索引擎

🚀 快速使用
基础重排序示例——python运行
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Rerank-M3 检索重排序示例"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple

# 环境配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen-Rerank-M3",
    trust_remote_code=True,
    cache_dir="./cache"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "./Qwen-Rerank-M3",
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE,
    cache_dir="./cache"
).to(DEVICE).eval()

# 重排序核心函数
def rerank_documents(
    query: str,
    candidates: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    对检索结果进行重排序
    :param query: 查询语句
    :param candidates: 候选文档列表
    :param top_k: 返回TOP-K结果
    :return: 排序后的(文档, 分数)列表
    """
    # 构造query-candidate对
    pairs = [[query, cand] for cand in candidates]
    
    # 文本编码
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)
    
    # 预测相关性分数
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).tolist()
    
    # 按分数降序排序
    ranked_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # 返回TOP-K
    return ranked_pairs[:top_k]

# 示例调用
if __name__ == "__main__":
    # 检索场景示例
    query = "如何在本地部署Qwen-2-7B-Chat并开启API服务"
    # 模拟检索结果
    retrieved_docs = [
        "Qwen-2模型可通过FastChat部署OpenAI兼容API，端口可自定义",
        "Qwen-Embedding-V2用于文本向量生成，适配RAG场景",
        "部署Qwen-2-7B-Chat需要安装torch、transformers等依赖，支持4bit量化",
        "Qwen-Coder-2可生成Python代码，支持代码调试功能",
        "FastChat支持多模型部署，包括Qwen、Llama、ChatGLM等"
    ]
    
    # 重排序
    ranked_results = rerank_documents(query, retrieved_docs, top_k=3)
    
    # 输出结果
    print(f"查询：{query}\n")
    print("重排序结果（相关性从高到低）：")
    for idx, (doc, score) in enumerate(ranked_results, 1):
        print(f"{idx}. 得分：{score:.4f}")
        print(f"   文本：{doc}\n")

RAG 集成示例（生产级）——python运行
# 与检索引擎集成示例
def rag_pipeline(query: str, top_k: int = 3) -> str:
    """
    RAG完整流程：检索 -> 重排序 -> 生成
    """
    # 1. 第一步：从检索引擎获取候选文档（模拟）
    retrieved_docs = retrieve_documents(query, top_k=10)
    
    # 2. 第二步：重排序
    ranked_docs = rerank_documents(query, retrieved_docs, top_k=top_k)
    
    # 3. 第三步：构造prompt并调用LLM生成回答
    context = "\n".join([doc for doc, _ in ranked_docs])
    prompt = f"""基于以下上下文回答问题：
{context}

问题：{query}
回答："""
    
    # 调用Qwen-2生成回答
    response = generate_answer(prompt)
    return response

