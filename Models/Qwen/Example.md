
###一、大模型
```
### Qwen3-30B-A3B-Instruct-2507-FP8 vllm启动命令（vllm=0.13.0）
nohup python3 -m vllm.entrypoints.openai.api_server \
--trust-remote-code \
--dtpye bfloat16 \
--model /qwen3/Qwen3-30B-A3B-Instruct-2507-FP8 \
--port 9000 \
--gpu-memory-utilization 0.95 \
--tensor-parallel-size 4 \
--max-model-len 128000 \
--max-num-seqs 50 \
--quantization fp8 \
--enable-auto-tool-choice \
--tool-call-parser hermes \
--served-model-name qwen3 \
--enforce-eager > Qwen3-30B-A3B-Instruct-2507-FP8.log 2>&1 &
```
###二、向量模型
```
###Qwen3-Embedding-8B vllm启动命令
nohup python3 -m vllm.entrypoints.openai.api_server \
--trust-remote-code \
--dtpye bfloat16 \
--model /qwen3/Qwen3-Embedding-8B \
--port 9000 \
--gpu-memory-utilization 0.95 \
--tensor-parallel-size 4 \
--max-num-batched-toens 40960 \
--max-num-seqs 50 \
--enable-log-requests
--served-model-name qwen3-embedding \
--enforce-eager > Qwen3-Embedding-8B.log 2>&1 &
```
###三、VL图像模型
```
###Qwen3-VL-30B-Instruct-2507  vllm启动命令
nohup python3 -m vllm.entrypoints.openai.api_server \
--trust-remote-code \
--dtpye bfloat16 \
--model /qwen3/Qwen3-VL-30B-Instruct-2507 \
--port 9000 \
--gpu-memory-utilization 0.95 \
--tensor-parallel-size 4 \
--max-model-len 128000 \
--max-num-seqs 50 \
--max-logprobs 20 \
--served-model-name qwen3 \
--enforce-eager > Qwen3-VL-30B-Instruct-2507.log 2>&1 &
```

