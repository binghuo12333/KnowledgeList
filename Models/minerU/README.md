### MinerU模型（文档理解多模态模型）

MinerU是专为复杂文档理解设计的轻量级视觉语言模型（VLM），聚焦OCR、表格解析、图表分析、文档问答等任务，以“小而精”的特点适配低成本部署场景。

### 核心特性

- **轻量高效**：主流版本（MinerU2.5-2509-1.2B）仅12亿参数，CPU即可运行，无需高端GPU。

- **文档专用优化**：配备专用视觉编码器，支持印刷体/手写体识别、版面结构感知（标题、段落、表格分层）。

- **多任务能力**：集成OCR、表格结构化解析、图表趋势分析、内容摘要、多轮问答于一体。

- **易用性强**：自带WebUI，支持文档图片上传、可视化交互，非技术人员可快速上手。

### 核心功能与应用场景

|功能模块|能力描述|应用示例|
|---|---|---|
|高精度OCR|提取印刷体/手写体文字，保留段落结构与阅读顺序|扫描版产品说明书文字提取|
|表格解析|结构化还原表格数据，支持错行修正|产品规格表转换为CSV格式|
|图表理解|分析柱状图、折线图趋势，解读坐标轴含义与数据关系|销售报表自动分析与总结|
|多轮问答|支持上下文记忆，连续追问文档相关问题|“该部件的工作温度是多少？”|
|内容摘要|自动生成文档核心要点，区分标题、正文、警告信息|技术手册关键操作步骤提炼|
### 实操使用指南

#### 1. 模型部署（基于CSDN星图平台镜像）

```bash

# 拉取MinerU镜像（需提前注册CSDN星图平台账号）
docker pull registry.cn-beijing.aliyuncs.com/csdn-star/mineu:2.5-2509-1.2B

# 启动容器，映射端口与数据目录
docker run -d \
  --name mineu-service \
  -p 7860:7860 \
  -v ./mineu-data:/app/data \
  registry.cn-beijing.aliyuncs.com/csdn-star/mineu:2.5-2509-1.2B

# 访问WebUI：http://localhost:7860
```

#### 2. 基础操作流程（WebUI）

1. 上传文档：点击“上传图片”，支持单张/多张文档图片（扫描件、截图均可），支持JPG、PNG、PDF格式。

2. 功能选择：根据需求选择“OCR提取”“表格解析”“图表分析”“问答交互”等功能。

3. 交互提问：在输入框中发起问题，例如“提取这张表格的数据并导出CSV”“分析图表中的销售额趋势”。

4. 结果导出：支持识别文本、表格数据、摘要内容导出为TXT、CSV格式。

#### 3. 进阶使用：提示词优化技巧

通过精准提示词提升模型效果，示例如下：

- 表格解析：“识别图片中的表格，修正错行数据，按行优先顺序导出为CSV格式，标注表头。”

- 图表分析：“识别这张折线图的坐标轴含义、数据节点，分析2023年四个季度的变化趋势，给出结论。”

- 内容摘要：“提取该产品说明书的核心功能、操作步骤与注意事项，分点总结，忽略无关广告内容。”

#### 4. 二次开发（API调用）

```python

import requests
import base64

# 读取图片并编码为base64
with open("document.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

# 调用MinerU API
response = requests.post(
    "http://localhost:7860/api/analyze",
    json={
        "image": img_base64,
        "task_type": "qa",  # 可选：ocr/table/chart/qa/summary
        "prompt": "这个产品的核心功能有哪些？"
    }
)

# 解析结果
result = response.json()
print(f"回答：{result['answer']}")
if "table_data" in result:
    print("表格数据：", result["table_data"])
```
