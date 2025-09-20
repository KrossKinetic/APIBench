#!/usr/bin/env python3
"""
Reads a CSV with columns including:
  ['id', 'Instruction', 'Results', 'rank_0', ..., 'rank_9']

Sends prompts to a vLLM OpenAI-compatible server via /v1/completions
(using the OpenAI SDK), and writes a results CSV including ModelResponse.

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_generation.py \
    --csv output.csv \
    --out other_output.csv \
    --server http://localhost:8000 \
    --model Qwen/Qwen2.5-Omni-3B \
    --max-tokens 512 \
    --temperature 0.2 \
    --seed 42

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Omni-3B --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --trust-remote-code
Used this to run server
"""

from __future__ import annotations
import argparse
import os
import time
from typing import List, Optional
import multiprocessing
from functools import partial

import pandas as pd
from openai import OpenAI, APIError, APITimeoutError, RateLimitError


def _normalize_ctx(x) -> str:
    """Turn NaN/None into empty string; ensure str."""
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def build_prompt(instruction: str, contexts: list[str]) -> str:
    # Label and join context docs (skip empties)
    ctx_labeled = [c.strip() for c in contexts if str(c).strip()]
    context_block = "\n".join(ctx_labeled)

    return (f"""
        You are an expert Python API recommendation system.  
        Given the OriginalQuery and the provided context, predict the most relevant Python APIs.  

        Requirements:  
        - Output only API names as a JSON array.  
        - No explanations or extra text.  
        - Format strictly as: ["api1()", "api2()", ...]  
        - Wrap the output in a ```json``` code block.  

        OriginalQuery: {instruction}
        Context: {context_block}
        """
    )



def call_vllm_completions_openai(
    prompt: str,
    server: str,
    model: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    stop: Optional[List[str]] = None,
    retries: int = 3,
    timeout: int = 120,
    api_key: Optional[str] = None,
) -> str:
    """
    Calls vLLM's OpenAI-compatible /v1/completions endpoint via OpenAI SDK.
    """
    client = OpenAI(
        base_url=server.rstrip("/") + "/v1",
        api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),  # vLLM accepts any non-empty key
        timeout=timeout,
    )
    backoff = 1.0
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,   # vLLM supports seed for determinism
                n=1,
                stop=stop,
            )
            return resp.choices[0].text
        except (APIError, APITimeoutError, RateLimitError, Exception) as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    raise RuntimeError(f"vLLM request failed after {retries} attempts: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with Autocompletion, Instruction, rank_0..rank_9")
    ap.add_argument("--out", required=True, help="Output CSV with ModelResponse column")
    ap.add_argument("--server", default="http://localhost:8000", help="vLLM base URL")
    ap.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Only evaluate first N rows (0 = all)")
    ap.add_argument("--stop-fence", action="store_true",
                    help="Add ``` as a stop sequence to trim trailing fences.")
    ap.add_argument("--api-key", default=None,
                    help="Optional API key to send to vLLM (any non-empty string works).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Validate core columns
    for col in ("Instruction"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.csv}")

    # It’s okay if some ranks are missing; we’ll treat them as empty strings.
    rank_cols = [f"rank_{i}" for i in range(10)]
    for rc in rank_cols:
        if rc not in df.columns:
            df[rc] = ""  # add missing rank columns as empty

    if args.limit > 0:
        df = df.head(args.limit).copy()

    stop = ["```"] if args.stop_fence else None

    outputs: List[str] = []
    
    # Changed the stuff below to allow for parallelization

    prompts = []
    for _, row in df.iterrows():
        contexts = [_normalize_ctx(row.get(f"rank_{i}", "")) for i in range(10)]
        prompt = build_prompt(
            instruction=_normalize_ctx(row["Instruction"]),
            contexts=contexts,
        )
        prompts.append(prompt)

    partial_worker = partial(call_vllm_completions_openai,
                    server=args.server,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    stop=stop,
                    api_key=args.api_key)
            
    with multiprocessing.Pool(processes=8) as pool:
        text = pool.map(partial_worker, prompts)

    # Changed the stuff above to allow for parallelization

    outputs = text

    out_df = df.copy()
    out_df["ModelResponse"] = outputs
    out_df = out_df[['Instruction', 'Test','ModelResponse']]
    out_df.to_csv(args.out, index=False)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    print(f"Execution time: {time.monotonic() - start_time:.4f} seconds")