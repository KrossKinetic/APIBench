import csv
import json
import re
from typing import List, Optional

API_TOKEN_RE = re.compile(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b')  # matches dotted names like pkg.mod.func

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
    # Try JSON first
    try:
        parsed = json.loads(rf)
        if isinstance(parsed, list):
            return [normalize_api(str(x)) for x in parsed if str(x).strip()]
    except Exception:
        pass

    # If not JSON, split on semicolon or comma (but keep things like "pkg.func, other" robust)
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
        headers = {h.lower(): h for h in reader.fieldnames}
        results_col = headers.get("results", "Results")
        response_col = headers.get("modelresponse", "ModelResponse")

        for row in reader:
            row_count += 1
            expected_raw = row.get(results_col, "") or ""
            response_raw = row.get(response_col, "") or ""

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

    print(f"\nMatched APIs: {matched_apis}/{total_apis}")
    print(f"Accuracy: {matched_apis / total_apis:.2%}")

if __name__ == "__main__":
    evaluate_api_hits("other_output.csv")
