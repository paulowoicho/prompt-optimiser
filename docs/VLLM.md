# Serving open models locally with vLLM

The library talks to any OpenAI-compatible endpoint through `VLLM(model=..., base_url=...)`.
This page records the exact commands used for the experiments in this repo so they can be
reproduced. vLLM lives in its own environment (`.venv-vllm`, vLLM 0.10.2) because its CUDA
and torch pins conflict with the library's optional extras.

## Machine

Eight GPUs: six RTX 6000 Ada (48 GB each, indices 0, 1, 4, 5, 6, 7) and two RTX 3090
(24 GB, indices 2 and 3; index 3 is in use by another process). Models are read from the
local Hugging Face cache with `HF_HUB_OFFLINE=1`, so nothing is downloaded at serve time.

## Environment

```bash
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python 'vllm==0.10.2' \
  'transformers==4.55.2' 'tokenizers==0.21.4' 'huggingface-hub==0.36.2' 'openai==1.109.1'
```

The pins matter: an unconstrained install pulled transformers 5, which failed to serve.

## Models served

| Served name | Hugging Face id | Weights | GPUs | Port |
|---|---|---|---|---|
| `qwen2.5-1.5b-instruct` | `Qwen/Qwen2.5-1.5B-Instruct` (copied to `runs/models/`) | 3 GB | one | 8123 |
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 15 GB | one Ada | 8124 |
| `qwen2.5-72b-instruct-awq` | `Qwen/Qwen2.5-72B-Instruct-AWQ` | 39 GB (4-bit) | two Adas, tensor-parallel | 8123 |

The 1.5B and 72B models share port 8123 because they were served at different times.
Only one server per port.

### Qwen2.5-72B-Instruct-AWQ, tensor-parallel across two RTX 6000 Ada

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ --served-model-name qwen2.5-72b-instruct-awq \
  --host 127.0.0.1 --port 8123 --tensor-parallel-size 2 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 16 --disable-log-requests \
  > logs/vllm-qwen72b-awq.log 2>&1 &
```

### Llama-3.1-8B-Instruct, one RTX 6000 Ada

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct --served-model-name llama-3.1-8b-instruct \
  --host 127.0.0.1 --port 8124 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 16 --disable-log-requests \
  > logs/vllm-llama8b.log 2>&1 &
```

### Qwen2.5-1.5B-Instruct, the original small model

```bash
.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model runs/models/qwen2.5-1.5b-instruct --served-model-name qwen2.5-1.5b-instruct \
  --host 127.0.0.1 --port 8123 --max-model-len 8192 --gpu-memory-utilization 0.35 \
  --max-num-seqs 8 --enforce-eager --disable-log-requests
```

## Check a server

```bash
curl -s http://127.0.0.1:8123/v1/models
curl -s http://127.0.0.1:8123/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen2.5-72b-instruct-awq",
  "messages": [{"role": "system", "content": "Reply with one word."}, {"role": "user", "content": "ham or spam: WIN A FREE PRIZE NOW"}],
  "max_tokens": 5, "temperature": 0}'
```

## Point the library at it

```python
from prompt_optimiser import VLLM

model = VLLM(model="qwen2.5-72b-instruct-awq", base_url="http://127.0.0.1:8123/v1", max_tokens=1024)
```

Authentication, if the server was started with `--api-key`, is read from the environment
variable named by `api_key_env` (default `VLLM_API_KEY`). The key is never written to
`config.json` or trackers.

## Stop a server

```bash
pkill -f 'served-model-name qwen2.5-72b-instruct-awq'
```
