# DeepSeek（深度求索）系列模型使用指南

<div align="center">
  <img src="https://raw.githubusercontent.com/deepseek-ai/DeepSeek-LLM/main/assets/deepseek-logo.png" width="180" alt="DeepSeek Logo">
  <p align="center"><strong>深度求索全系列开源模型部署与使用指南</strong></p>
  
  <!-- 徽章集合 -->
  <a href="https://github.com/deepseek-ai/DeepSeek-LLM" target="_blank">
    <img src="https://img.shields.io/github/stars/deepseek-ai/DeepSeek-LLM?style=social" alt="GitHub Stars">
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-DeepSeek-yellow" alt="HuggingFace">
  </a>
  <a href="https://modelscope.cn/organization/deepseek-ai" target="_blank">
    <img src="https://img.shields.io/badge/ModelScope-DeepSeek-ff69b4" alt="ModelScope">
  </a>
  <a href="https://www.deepseek.com/" target="_blank">
    <img src="https://img.shields.io/badge/官网-DeepSeek-blue" alt="DeepSeek Official">
  </a>
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</div>

---

## 🔗 官方资源

| 平台 | 链接 | 核心用途 |
| :--- | :--- | :--- |
| 官方 GitHub | [https://github.com/deepseek-ai](https://github.com/deepseek-ai) | 源码、部署脚本、示例代码 |
| 官方网站 | [https://www.deepseek.com/](https://www.deepseek.com/) | 产品介绍、API 文档、商用说明 |
| Hugging Face | [https://huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai) | 模型下载、HF 生态适配 |
| 魔搭社区 | [https://modelscope.cn/organization/deepseek-ai](https://modelscope.cn/organization/deepseek-ai) | 中文生态、一键部署、国内加速 |
| 技术文档 | [https://docs.deepseek.com/](https://docs.deepseek.com/) | 详细部署指南、API 使用说明 |

---

## 📚 模型分类及使用说明

### 1. 通用大语言模型（DeepSeek-LLM）
> **核心特性**：支持 64K 超长上下文，中文理解/推理能力突出，开源 7B/16B/33B 版本，商用友好

#### 📋 最新模型列表
| 模型名称 | 规模 | 上下文长度 | 核心优势 | 显存要求（INT4） |
| :------- | :--- | :--------- | :------- | :--------------- |
| `deepseek-llm-7b-chat` | 7B | 64K | 轻量高效，消费级显卡可运行 | 6-8GB |
| `deepseek-llm-16b-chat` | 16B | 64K | 平衡版，性能与资源兼顾 | 12-14GB |
| `deepseek-llm-33b-chat` | 33B | 64K | 旗舰版，推理能力更强 | 24-26GB |
| `deepseek-llm-7b-base` | 7B | 64K | 基础版，适合二次微调 | 6-8GB |

#### 🚀 快速启动
##### 环境准备
```bash
# 安装核心依赖
pip install "fschat[model_worker,webui]" transformers torch accelerate sentencepiece protobuf
```
# 1. 命令行交互（INT4 量化）
```
# DeepSeek-7B-Chat 命令行对话
python -m fastchat.serve.cli \
  --model-path ./deepseek-llm-7b-chat \
  --load-4bit \
  --trust-remote-code \
  --temperature 0.7 \
  --max-new-tokens 2048

# 超长上下文测试（64K）
python -m fastchat.serve.cli \
  --model-path ./deepseek-llm-7b-chat \
  --load-4bit \
  --trust-remote-code \
  --max-context-length 65536 \
  --max-new-tokens 1024
```
# 2. WebUI 可视化服务
```
# 启动控制器（后台运行）
nohup python -m fastchat.serve.controller > controller.log 2>&1 &
# 启动模型 Worker
nohup python -m fastchat.serve.model_worker \
  --model-path ./deepseek-llm-7b-chat \
  --device cuda \
  --load-4bit \
  --trust-remote-code > model_worker.log 2>&1 &

# 启动 WebUI（访问：http://localhost:7860）
nohup python -m fastchat.serve.gradio_web_server > webui.log 2>&1 &
```
# 3. OpenAI 兼容 API 服务
```
# 启动 API 服务（端口 8000）
python -m fastchat.serve.openai_api_server \
  --model-path ./deepseek-llm-7b-chat \
  --host 0.0.0.0 \
  --port 8000 \
  --load-4bit \
  --trust-remote-code

# API 调用示例（curl）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-llm-7b-chat",
    "messages": [{"role": "user", "content": "介绍一下 DeepSeek 大模型的核心优势"}]
  }'
```
### 2. 多模态模型（DeepSeek-VL）
🚀 快速启动
环境准备
# 安装多模态依赖
```
pip install torchvision pillow transformers accelerate opencv-python
```
# 1. 命令行图文对话
```
# 图文问答（指定本地图片）
python -m fastchat.serve.cli \
  --model-path ./deepseek-vl-7b-chat \
  --load-4bit \
  --trust-remote-code \
  --image ./test_image.jpg \
  --temperature 0.7

# 示例提问："分析这张图表的数据趋势，并给出结论"
```
# 2. WebUI 多模态服务
```
# 启动控制器
python -m fastchat.serve.controller &

# 启动多模态 Worker
python -m fastchat.serve.model_worker \
  --model-path ./deepseek-vl-7b-chat \
  --device cuda \
  --load-4bit \
  --trust-remote-code &

# 启动带图片上传的 WebUI
python -m fastchat.serve.gradio_web_server --multimodal
```
### 3. 嵌入模型（DeepSeek-Embedding）
核心特性：支持中英文双语嵌入，64K 长文本嵌入，适配 RAG / 检索 / 聚类场景，向量维度 1024
📋 最新模型列表
| 模型名称 | 向量维度 | 上下文长度 | 核心优势 | 显存要求（INT4） |
| :------- | :--- | :--------- | :------- | :--------------- |
| `deepseek-embedding-v1` | 1024 | 64K | 通用文本嵌入 | ≤2GB |
| `deepseek-embedding-long-context` | 1024 | 64K | 长文本专用 | ≤3GB |

🚀 快速使用
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek-Embedding 文本嵌入示例"""
from transformers import AutoModel, AutoTokenizer
import torch

# 环境配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(
    "./deepseek-embedding-v1",
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "./deepseek-embedding-v1",
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE
).to(DEVICE).eval()

# 生成文本嵌入
def get_embedding(texts: list) -> torch.Tensor:
    """
    生成文本嵌入向量（归一化）
    :param texts: 文本列表
    :return: 嵌入向量 [batch_size, 1024]
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=65536,  # 64K 上下文
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 取 <bos> token 作为句子嵌入
        embeddings = outputs.last_hidden_state[:, 0]
        # 向量归一化（检索场景必做）
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings

# 示例调用
if __name__ == "__main__":
    test_texts = [
        "DeepSeek 大模型支持 64K 超长上下文",
        "DeepSeek-Embedding 适配检索问答场景",
        "DeepSeek is an open-source LLM with 64K context window"
    ]
    
    embeddings = get_embedding(test_texts)
    print(f"嵌入向量维度：{embeddings.shape}")  # torch.Size([3, 1024])
    
    # 计算相似度
    sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    print(f"文本相似度：{sim.item():.4f}")
```

### 4. 代码模型（DeepSeek-Coder）
核心特性：支持 80+ 编程语言，128K 代码上下文，代码生成 / 补全 / 调试 / 重构，适配专业开发场景
📋 最新模型列表
| 模型名称 | 规模 | 上下文长度 | 核心优势 | 显存要求（INT4） |
| :------- | :--- | :--------- | :------- | :--------------- |
| `deepseek-coder-7b-instruct` | 7B | 128K | 基础代码开发 | 6-8GB |
| `deepseek-coder-16b-instruct` | 16B | 128K | 复杂代码开发 | 12-14GB |
| `deepseek-coder-33b-instruct` | 33B | 128K | 旗舰代码模型 | 24-26GB |
| `deepseek-coder-v2-7b` | 7B | 128K | 第二代代码模型 | 6-8GB |

🚀 快速启动
1. 命令行代码对话
```
# 代码开发（INT4 量化）
python -m fastchat.serve.cli \
  --model-path ./deepseek-coder-7b-instruct \
  --load-4bit \
  --trust-remote-code \
  --prompt-template deepseek_coder \
  --temperature 0.2 \
  --max-new-tokens 2048

2. 代码补全 API 服务
# 启动代码补全 API
python -m fastchat.serve.openai_api_server \
  --model-path ./deepseek-coder-7b-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --load-4bit \
  --trust-remote-code

# 代码补全调用示例
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-7b-instruct",
    "prompt": "def quick_sort(arr):\n    # 快速排序实现",
    "max_tokens": 512,
    "temperature": 0.1
  }'
```

### 5. 重排序模型（DeepSeek-Rerank）
核心特性：检索问答（RAG）专用，支持中英文重排、长文本重排，单条推理耗时 <10ms
📋 最新模型列表
| 模型名称 | 规模 | 上下文长度 | 核心优势 | 显存要求（INT4） |
| :------- | :--- | :--------- | :------- | :--------------- |
| `deepseek-rerank-base` | 1.3B | 512 | 通用重排 | ≤2GB |
| `deepseek-rerank-large` | 2.6B | 1024 | 高精度重排 | ≤4GB |
| `deepseek-rerank-long` | 2.6B | 2048 | 长文本重排 | ≤4GB |

🚀 快速使用
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek-Rerank 检索重排序示例"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple

# 环境配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(
    "./deepseek-rerank-base",
    trust_remote_code=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    "./deepseek-rerank-base",
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE
).to(DEVICE).eval()

# 重排序核心函数
def rerank_docs(
    query: str,
    candidates: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    检索结果重排序
    :param query: 查询语句
    :param candidates: 候选文档列表
    :param top_k: 返回 TOP-K 结果
    :return: (文档, 分数) 列表
    """
    # 构造 query-candidate 对
    pairs = [[query, doc] for doc in candidates]
    
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
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# 示例调用
if __name__ == "__main__":
    query = "如何部署 DeepSeek-7B-Chat 并开启 64K 上下文"
    candidates = [
        "DeepSeek-LLM 支持 64K 超长上下文，部署时需设置 max_context_length",
        "DeepSeek-Embedding 用于文本向量生成，适配 RAG 场景",
        "部署 DeepSeek-7B-Chat 需要安装 torch>=2.0.0 和 transformers>=4.35.0",
        "DeepSeek-Coder 支持 128K 代码上下文，适合专业开发"
    ]
    
    # 重排序
    results = rerank_docs(query, candidates, top_k=3)
    print("重排序结果（相关性从高到低）：")
    for idx, (doc, score) in enumerate(results, 1):
        print(f"{idx}. 得分：{score:.4f} | 文本：{doc}")
```

📝 重要注意事项
# 1. 模型下载
```
# 方法 1：魔搭下载（国内推荐）
pip install modelscope
modelscope download --model=deepseek-ai/deepseek-llm-7b-chat --local-dir=./deepseek-llm-7b-chat

# 方法 2：Hugging Face 下载
pip install huggingface-hub
huggingface-cli download deepseek-ai/deepseek-llm-7b-chat --local-dir ./deepseek-llm-7b-chat
```
# 2. 关键部署技巧
超长上下文：启动时添加 --max-context-length 65536 开启 64K 上下文
量化启动：低显存场景必加 --load-4bit/--load-8bit，性能损失 <5%
依赖兼容：建议使用 torch>=2.0.0、transformers>=4.35.0
商用说明：DeepSeek 系列遵循 MIT 协议，可免费商用（需保留版权声明）
# 3. 常见问题
64K 上下文启动报错：需升级 transformers 到 4.35.0+
多模态模型图片加载失败：安装 pillow>=10.0.0、opencv-python>=4.8.0
量化启动失败：安装 bitsandbytes>=0.41.0
