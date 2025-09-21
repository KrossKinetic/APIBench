import csv
import json
import re
from typing import List, Optional
import os
from openai import OpenAI
import argparse

API_TOKEN_RE = re.compile(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b')

def normalize_api(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # remove trailing parentheses "()", possibly with spaces " ( ) "
    s = re.sub(r'\(\s*\)$', '', s)
    # remove trailing commas/semicolons and whitespace
    s = s.rstrip(';, ')
    return s

def parse_expected_list(results_field: str) -> List[str]:
    """
    Parse the Results column which may be semicolon- or comma-separated,
    or might already be a JSON array string.
    """
    if results_field is None:
        return []
    rf = results_field.strip()
    
    try:
        parsed = json.loads(rf)
        if isinstance(parsed, list):
            return [normalize_api(str(x)) for x in parsed if str(x).strip()]
    except Exception:
        pass

    parts = re.split(r'[;,]\s*', rf)
    parts = [normalize_api(p) for p in parts if p and p.strip()]
    return parts

def extract_json_arrays(text: str) -> Optional[List[str]]:
    """
    Find candidate JSON array substrings like [...] inside text and try to parse them.
    Return first successfully parsed list-of-strings, or None.
    """
    if text is None:
        return None
    # remove surrounding code fences (one or more)
    cleaned = re.sub(r'```[^\n]*\n', '', text)
    cleaned = cleaned.replace('```', '')
    # find bracketed arrays
    candidates = re.findall(r'(\[.*?\])', cleaned, flags=re.DOTALL)
    for cand in candidates:
        # Try some quick fixes: replace doubled double-quotes that sometimes appear in CSV dumps
        attempt = cand.replace('""', '"')
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return [normalize_api(str(x)) for x in parsed if str(x).strip()]
        except Exception:
            # try single-quoted JSON -> convert to double quotes naively
            attempt2 = re.sub(r"(?<!\\)\'", '"', attempt)
            try:
                parsed = json.loads(attempt2)
                if isinstance(parsed, list):
                    return [normalize_api(str(x)) for x in parsed if str(x).strip()]
            except Exception:
                pass
    return None

def extract_predicted_from_text(text: str) -> List[str]:
    """
    Try JSON arrays -> else fallback to regex to extract dotted API-like tokens.
    """
    js = extract_json_arrays(text)
    if js is not None:
        return js
    if text is None:
        return []
    # fallback: find dotted identifiers like numpy.random.randint
    found = API_TOKEN_RE.findall(text)
    found = [normalize_api(f) for f in found]
    # remove duplicates while preserving order
    seen = set()
    out = []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def evaluate_api_hits(csv_file: str):
    total_apis = 0
    matched_apis = 0
    row_count = 0

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row_count = 0

        for row in reader:
            row_count += 1

            expected_raw = row["Results"]
            response_raw = row["ModelResponse"]
            query = row["Instruction"]

            expected_list = parse_expected_list(expected_raw)
            predicted_list = extract_predicted_from_text(response_raw)

            total_apis += len(expected_list)
            for api in expected_list:
                matched = False
                for p in predicted_list:
                    if not api:
                        continue
                    if api == p or api in p or p in api:
                        matched = True
                        break
                if matched:
                    matched_apis += 1
                else:
                    if LLM_match("; ".join(predicted_list),api,query):
                        matched_apis += 1

    print(f"\nMatched APIs: {matched_apis}/{total_apis}")
    print(f"Accuracy: {matched_apis / total_apis:.2%}")

def LLM_match(predicted_list_str: str, api: str, query: str) -> bool:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""
    You are evaluating API recommendation results.

    Query: {query}

    Ground truth API (may be noisy or ambiguous): {api}

    Model predicted APIs: {predicted_list_str}

    Task: Decide if the model's predicted APIs provide a correct and reasonable solution to the query, 
    even if the ground truth is wrong. Return only a single word:
    True  → if correct
    False → if not correct
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3
    )

    try:
        result = response.choices[0].message.content.strip()
        return result.lower().startswith("true")
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", nargs="?", default="other_output.csv", help="Path to input CSV file with results to evaluate")
    args = parser.parse_args()

    evaluate_api_hits(args.input_csv)

