import os
import json
import ast
import argparse
import time
from openai import OpenAI
from typing import List, Dict, Any, Optional

import pandas as pd

def parse_sections(sections_raw: Any) -> List[Dict[str, Any]]:
    """
    Parse the 'sections' field safely from CSV.
    Accepts:
      - JSON string
      - Python literal string (list[dict])
      - Already-parsed list[dict]
      - NaN/None -> empty list
    """
    if sections_raw is None or (isinstance(sections_raw, float) and pd.isna(sections_raw)):
        return []

    if isinstance(sections_raw, list):
        return sections_raw

    if isinstance(sections_raw, str):
        s = sections_raw.strip()
        if not s:
            return []
        # Try JSON first
        try:
            val = json.loads(s)
            return val if isinstance(val, list) else []
        except Exception:
            pass
        # Try Python literal (e.g., "['x', {'y': 1}]")
        try:
            val = ast.literal_eval(s)
            return val if isinstance(val, list) else []
        except Exception:
            return []

    return []

def extract_key_concepts_from_sections(sections: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all 'key_concept' values from a list of section dicts.
    Handles:
      - 'key_concept' as string
      - 'key_concept' as list[str]
      - alternative key name 'key_concepts'
    Deduplicates and preserves original order.
    """
    concepts: List[str] = []
    seen = set()

    for sec in sections:
        if not isinstance(sec, dict):
            continue

        # prefer 'key_concept', fallback to 'key_concepts'
        val = None
        if 'key_concept' in sec:
            val = sec.get('key_concept')
        elif 'key_concepts' in sec:
            val = sec.get('key_concepts')

        if not val:
            continue

        if isinstance(val, str):
            cleaned = val.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                concepts.append(cleaned)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned and cleaned.lower() not in seen:
                        seen.add(cleaned.lower())
                        concepts.append(cleaned)

    return concepts

def build_llm_prompt(course_code: str, course_name: str, file_name: str, key_concepts: List[str], extra_hint: Optional[str] = None) -> str:
    """
    Construct a concise prompt asking the LLM to output a short list of course topics.
    Output must be JSON array of short topic strings.
    """
    kc = ", ".join(key_concepts[:20]) if key_concepts else "(none)"
    hint = f"\nAdditional hint: {extra_hint}" if extra_hint else ""
    prompt = f"""
You are classifying the topics covered by a course file for course {course_name}.

Course code: {course_code}
Course name: {course_name}

Key concepts extracted from file sections: {kc}{hint}

Return a concise JSON array of topic tags (no explanations) that best describe the main topics of the file named '{file_name}'.
Rules:
- Only output a JSON array (no explanation).
- Use lower case phrases, 1-4 words each.
- Prefer CS 61A canonical topic names if applicable.
"""
    return prompt.strip()

def call_llm(prompt: str, model: str = "openai/gpt-oss-120b", provider: str = "groq", max_retries: int = 3, sleep_s: float = 2.0) -> List[str]:
    """
    Call LLM via OpenAI-compatible SDK.
    - provider='groq' uses GROQ_API_KEY and base_url https://api.groq.com/openai/v1
    - provider='openai' uses OPENAI_API_KEY and default base_url
    Returns a list[str] topics. Falls back to [] on errors.
    """

    client = None
    if provider.lower() == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return []
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        # Map any OpenAI-only defaults to a Groq model
        
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []
        client = OpenAI(api_key=api_key)

    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise classifier that outputs only valid JSON arrays of short topic tags."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            topics = json.loads(content)
            if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                cleaned = []
                seen = set()
                for t in topics:
                    c = t.strip().lower()
                    if c and c not in seen:
                        seen.add(c)
                        cleaned.append(c)
                return cleaned
            return []
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    return []

def heuristic_topics(file_name: str, key_concepts: List[str]) -> List[str]:
    """
    Offline fallback: infer topics from filename and key concepts.
    Returns up to ~6 tags.
    """
    text = f"{file_name} " + " ".join(key_concepts or [])
    t = text.lower()

    rules = [
        ("higher-order functions", ["higher-order", "hof", "lambda", "map", "filter", "reduce"]),
        ("recursion", ["recursion", "recursive", "tail recursion", "tree recursion"]),
        ("trees", ["tree", "binary tree", "bst"]),
        ("linked lists", ["linked list", "node", "pointer"]),
        ("functions", ["function", "def", "call", "parameter", "argument"]),
        ("environment model", ["environment", "frame", "binding"]),
        ("oop", ["class", "object", "inheritance", "method", "attribute", "oop"]),
        ("iterators", ["iterator", "generator", "yield", "iter"]),
        ("scheme", ["scheme", "scm"]),
        ("sql", ["sql", "database", "select", "join"]),
        ("mutation", ["mutation", "mutate", "state", "assign", "set"]),
        ("complexity", ["big-o", "complexity", "runtime", "efficiency"]),
        ("sorting", ["sort", "merge sort", "quicksort", "heap sort"]),
        ("interpreters", ["interpreter", "eval", "parse"]),
        ("streams", ["stream", "lazy"]),
        ("data abstraction", ["abstraction", "adt", "interface"]),
        ("control", ["if", "else", "while", "for", "loop", "condition"]),
        ("dictionaries", ["dict", "dictionary", "hash map", "hashmap"]),
        ("tuples", ["tuple"]),
        ("lists", ["list", "list comprehension"]),
    ]

    found = []
    seen = set()
    for tag, kws in rules:
        if any(k in t for k in kws):
            if tag not in seen:
                seen.add(tag)
                found.append(tag)
        if len(found) >= 6:
            break

    # Also add top key_concepts terms if nothing found
    if not found and key_concepts:
        for kc in key_concepts[:4]:
            c = kc.strip().lower()
            if c and c not in seen:
                seen.add(c)
                found.append(c)

    return found[:6]

def process_csv(
    input_csv: str,
    output_csv: str,
    course_code: str = "CS 61A",
    course_name: str = "Structure and Interpretation of Computer Programs",
    use_llm: bool = True,
    model: str = "openai/gpt-oss-120b",
) -> pd.DataFrame:
    """
    - Loads input CSV
    - Adds 'key_concepts' column (JSON array string)
    - Adds 'topic' column (JSON array string) via LLM or heuristic fallback
    - Writes output CSV
    """
    abs_path = os.path.abspath(input_csv)
    df = pd.read_csv(abs_path)

    # Ensure sections column exists
    if "sections" not in df.columns:
        df["sections"] = None

    # Extract key_concepts list
    key_concepts_lists: List[List[str]] = []
    for _, row in df.iterrows():
        sections = parse_sections(row.get("sections"))
        concepts = extract_key_concepts_from_sections(sections)
        key_concepts_lists.append(concepts)

    df["key_concepts"] = [json.dumps(lst, ensure_ascii=False) for lst in key_concepts_lists]

    # Generate topics per row
    topics_col: List[str] = []
    for idx, row in df.iterrows():
        file_name = str(row.get("file_name") or row.get("filename") or row.get("name") or "")
        key_concepts = key_concepts_lists[idx]

        topics: List[str] = []
        if use_llm and os.getenv("OPENAI_API_KEY"):
            prompt = build_llm_prompt(course_code, course_name, file_name, key_concepts)
            topics = call_llm(prompt, model=model)

        if not topics:
            topics = heuristic_topics(file_name, key_concepts)

        topics_col.append(json.dumps(topics, ensure_ascii=False))

    df["topic"] = topics_col

    # Write output
    df.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv} (rows: {len(df)})")
    return df

def main():
    parser = argparse.ArgumentParser(description="Add key_concepts and topic columns to cs61a_files.csv")
    parser.add_argument("--input", "-i", default="cs61a_short.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="cs61a_files_with_topics.csv", help="Output CSV path")
    parser.add_argument("--course-code", default="CS 61A")
    parser.add_argument("--course-name", default="Structure and Interpretation of Computer Programs")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM; use heuristic only")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="OpenAI model name")
    args = parser.parse_args()

    use_llm = not args.no_llm
    process_csv(
        input_csv=args.input,
        output_csv=args.output,
        course_code=args.course_code,
        course_name=args.course_name,
        use_llm=use_llm,
        model=args.model,
    )


def find_metadata_db():
    """Find the metadata.db file in the project"""
    
    # Possible locations for cs61a_files.csv in the tai project
    possible_paths = [
        "../../cs61a_files.csv",                              # tai/cs61a_files.csv
        "../../ai_chatbot_backend/cs61a_files.csv",           # tai/ai_chatbot_backend/cs61a_files.csv
        "../../ai_chatbot_backend/data/cs61a_files.csv",      # tai/ai_chatbot_backend/data/cs61a_files.csv
        "../../../cs61a_files.csv",                           # in case of different structure
        "cs61a_files.csv"                                      # current directory
    ]

    print("Searching for cs61a_files.csv in tai project...")

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        print(f"Checking: {abs_path}")
        if os.path.exists(abs_path):
            print(f"FOUND: {abs_path}")
            print(path)
            df = pd.read_csv(abs_path)
            print(df)
            return abs_path
    
    print("\ncs61a_files.csv not found in expected locations.")
    print("Please check if the database exists by running:")
    print("  1. cd ../../ai_chatbot_backend")
    print("  2. make db-init  (or poetry run python scripts/initialize_db_and_files.py)")
    return None

if __name__ == "__main__":
    main()
    #find_metadata_db()