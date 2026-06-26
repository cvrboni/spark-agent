# Local Models

SparkAgent is optimized for local OpenAI-compatible servers, especially vLLM.

## vLLM DeepSeek Example

```bash
vllm serve deepseek-ai/DeepSeek-V4 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp"}' \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice
```

Then configure SparkAgent:

```bash
spark-agent init --base-url http://localhost:8000/v1 --model deepseek-v4-flash
spark-agent doctor
```

## Remote Local Network Node

For a vLLM server reachable over a private network:

```bash
spark-agent init \
  --base-url http://llm.local:8000/v1 \
  --model deepseek-v4-flash
```

If your gateway requires a bearer token:

```bash
export SPARK_AGENT_API_KEY=...
spark-agent doctor
```

## Language

SparkAgent defaults to `auto`:

- Italian prompt: Italian answer
- English prompt: English answer
- Code identifiers and tool outputs remain unchanged

Force a language with:

```bash
spark-agent init --language it --force
spark-agent init --language en --force
```
