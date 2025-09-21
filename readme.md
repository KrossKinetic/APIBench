# Retrieval Script (retrieval.py)

This script uses a SentenceTransformer model to embed a corpus and a set of queries, then retrieves the top-k most relevant documents for each query using semantic search.

## Usage
```bash
python retrieval.py \
  --model codesage/codesage-base-v2 \
  --input_corpus_csv corpus.csv \
  --input_query_json queries.json \
  --top_k 3 \
  --output_csv output.csv
```

# Run Generation with vLLM (generation.py)

This script takes a CSV of queries + retrieval results (from previous script), builds prompts, and queries a vLLM server (OpenAI-compatible API) to generate model responses. The output is a new CSV with model predictions.

## Usage
Start a vLLM server:
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Omni-3B --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.80 --trust-remote-code --quantization bitsandbytes
```

Run the script:
```bash
CUDA_VISIBLE_DEVICES=0 python generation.py \
  --csv output.csv \
  --out other_output.csv \
  --server http://localhost:8000 \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --max-tokens 512 \
  --temperature 0.2 \
  --seed 42
```

# API Evaluation with LLM Judge (check_results.py)

This script evaluates API recommendation results from a CSV file (previous result from other_output.csv).  
It compares model predictions against the ground truth APIs and uses an LLM (GPT-4o) as a fallback judge when the dataset is noisy or ambiguous.

## What it does
- Parses ground truth APIs from the **Results** column.
- Extracts predicted APIs from the **ModelResponse** column (JSON arrays or regex fallback).
- Counts matches directly (exact or substring match).
- If no direct match is found, asks GPT-4o to decide if the prediction is still reasonable given the query.
- Reports total matched APIs and overall accuracy.

## Usage
1. Make sure your environment has an OpenAI API key set:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. Run the script, by default it accepts (other_output.csv):
    ```bash
    python3 evaluate.py
    ```

# APIBench JSON → CSV Converter

This script converts the APIBench-Q dataset from JSON format into a simpler CSV format with three columns:
Query, APIs, and APIClasses for training purposes.

## Usage

```bash
python convert_to_csv.py input.json output.csv
```

- input.json (default: ./Python_Queries/OriginalPythonQueries.json)
- output.csv (default: ./Python_Queries/Python_Queries_Formatted.csv)

If no arguments are provided, the defaults will be used.
