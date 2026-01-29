"""
YouTube Lecture Summary JSON Generator
=======================================
Generates lecture topic summaries using OpenAI structured output API.

This script processes YouTube video lecture data to create comprehensive
topic summaries for each lecture, similar to lec_summary_json.py but
focusing on video transcript content.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import json
import pandas as pd
import re


class LectureSummary(BaseModel):
    """Structured output model for lecture summaries."""
    topic: str
    summary: str


def _safe_json_or_literal_load(s):
    """
    Try JSON first, then Python literal to parse a list/dict from a string.
    
    Args:
        s: String to parse (JSON or Python literal)
    
    Returns:
        Parsed object (dict/list) or None on failure
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if not isinstance(s, str):
        return s
    st = s.strip()
    if not st:
        return None
    try:
        return json.loads(st)
    except Exception:
        try:
            import ast
            return ast.literal_eval(st)
        except Exception:
            return None


def generate_openai_lecture_topics_json(
    summary_csv: str = "cs_61a_lecture_summary.csv",
    out_json: str = "cs_61a_youtube_lecture_topic_summaries.json",
    model: str = "gpt-4o-mini"
):
    """
    Create a JSON mapping of "Lecture X" -> {topic, summary, date} using OpenAI.
    
    Uses structured output (beta.chat.completions.parse) to ensure reliable
    JSON responses from the API.
    
    Args:
        summary_csv: Path to lecture summary CSV with aggregated concepts
        out_json: Output JSON file path
        model: OpenAI model to use for generation
    
    Returns:
        dict: The generated lecture topics dictionary, or None on error
    """
    # Path resolution - check multiple locations
    original_csv_path = summary_csv
    if not os.path.exists(summary_csv):
        # Try parent directories
        for levels_up in range(1, 4):
            potential_path = os.path.join("../" * levels_up, original_csv_path)
            if os.path.exists(potential_path):
                summary_csv = potential_path
                break

    if not os.path.exists(summary_csv):
        print(f"Error: Lecture summary CSV not found: {original_csv_path}")
        print(f"Searched locations:")
        print(f"  - Current directory: {os.path.abspath(original_csv_path)}")
        for levels_up in range(1, 4):
            print(f"  - {levels_up} level(s) up: {os.path.abspath(os.path.join('../' * levels_up, original_csv_path))}")
        return None

    print(f"Reading summary CSV: {summary_csv}")
    print(f"Absolute path: {os.path.abspath(summary_csv)}")
    
    df = pd.read_csv(summary_csv)
    
    if df.empty:
        print("Error: CSV is empty.")
        return None

    print(f"Loaded {len(df)} lecture records from CSV.")

    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = None
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client: {e}")
            client = None
    else:
        print("Error: No OPENAI_API_KEY found in environment.")
        print("Cannot generate summaries without API key.")
        return None

    results = {}
    
    for idx, row in df.iterrows():
        # Lecture numbering starts at 1
        lecture_key = f"Lecture {idx + 1}"
        
        # Extract data from CSV
        date = str(row.get('date', '') or '')
        topic = str(row.get('topic', '') or '')
        
        # Parse JSON/literal fields
        file_concepts_map_raw = row.get('file_concepts_map')
        file_descriptions_map_raw = row.get('file_descriptions_map')
        
        file_concepts_map = _safe_json_or_literal_load(file_concepts_map_raw) or {}
        file_descriptions_map = _safe_json_or_literal_load(file_descriptions_map_raw) or {}

        out_topic = ''
        out_summary = ''

        if client is not None:
            # Construct prompt for OpenAI
            prompt = (
                "Summarize a CS 61A lecture using grouped key concepts and aspects.\n\n"
                f"Lecture topic hint: {topic}\n\n"
                "Per-file Aggregated Concepts (key concepts and aspects):\n"
                f"{json.dumps(file_concepts_map, ensure_ascii=False, indent=2)}\n\n"
                "Per-file Descriptions (what is each file about):\n"
                f"{json.dumps(file_descriptions_map, ensure_ascii=False, indent=2)}\n\n"
                "Rules:\n"
                "- topic: 3-8 words describing the lecture's main focus.\n"
                "- summary: 2-4 sentences; emphasize central ideas and conceptual progression.\n"
                "- Avoid mentioning file names directly.\n"
                "- Consider whether the file content matches the lecture topic.\n"
                "- Focus on the educational concepts being taught.\n"
            )
            
            try:
                # Use structured output API for reliable JSON responses
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that summarizes CS course lectures."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        },
                    ],
                    response_format=LectureSummary,
                    temperature=0.3,
                    max_tokens=300,
                )
                
                parsed_response = completion.choices[0].message.parsed
                if parsed_response:
                    out_topic = parsed_response.topic.strip()
                    out_summary = parsed_response.summary.strip()
                    print(f"[+] {lecture_key}: {out_topic}")
                else:
                    print(f"[!] {lecture_key}: Structured output parsing failed/refused")
                    # Fallback to original topic
                    out_topic = topic

            except Exception as e:
                print(f"[X] {lecture_key}: OpenAI call failed - {e}")
                # Fallback to original topic
                out_topic = topic

        else:
            # No API client - use fallback
            out_topic = topic
            out_summary = "Summary unavailable (no API key)"

        # Store results
        results[lecture_key] = {
            # "date": date,
            "topic": out_topic if out_topic else topic,
            "summary": out_summary,
        }

    # Create output directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to output folder
    final_output_path = os.path.join(output_dir, os.path.basename(out_json))
    
    # Save to JSON
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Lecture topic summaries saved to: {final_output_path}")
    print(f"Absolute path: {os.path.abspath(final_output_path)}")
    print(f"Generated {len(results)} lecture summaries.")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    generate_openai_lecture_topics_json()
