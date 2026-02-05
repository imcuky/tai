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
    prediction_csv: str = "output/cs_61a_video_lecture_eval.csv",
    metadata_csv: str = "output/cs61a_youtube_files.csv",
    out_json: str = "cs_61a_youtube_lecture_topic_summaries.json",
    model: str = "gpt-4o-mini"
):
    """
    Create a JSON mapping of "Lecture X" -> {topic, summary} using OpenAI.

    Aggregates YouTube video content based on predictions.
    
    Args:
        prediction_csv: CSV with 'relative_path' and 'predicted_lecture'
        metadata_csv: CSV with 'relative_path', 'sections', 'description', 'file_name'
        out_json: Output JSON file path
        model: OpenAI model to use for generation
    """
    # 1. Load Predictions (File -> Lecture)
    pred_path = prediction_csv
    if not os.path.exists(pred_path):
        # Try finding it in the same directory as script if not found relative to cwd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pred_path = os.path.join(script_dir, prediction_csv)
    
    if not os.path.exists(pred_path):
        # Last resort: try just the filename in current dir
        pred_path = os.path.basename(prediction_csv)

    if not os.path.exists(pred_path):
        print(f"Error: Prediction CSV not found: {prediction_csv}")
        return None
        
    print(f"Reading predictions: {pred_path}")
    df_preds = pd.read_csv(pred_path)
    
    # 2. Load Metadata (File -> Content)
    meta_path = metadata_csv
    if not os.path.exists(meta_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        meta_path = os.path.join(script_dir, metadata_csv)
    
    if not os.path.exists(meta_path):
         meta_path = os.path.basename(metadata_csv)

    if not os.path.exists(meta_path):
        print(f"Error: Metadata CSV not found: {metadata_csv}")
        return None

    print(f"Reading metadata: {meta_path}")
    df_meta = pd.read_csv(meta_path)
    
    # 3. Merge on relative_path
    # Ensure relative_path columns match format if possible
    df = df_preds.merge(df_meta, on='relative_path', how='inner', suffixes=('', '_meta'))
    print(f"Merged data: {len(df)} records linked.")

    # 4. Filter for valid predictions (numeric > 0)
    df = df[pd.to_numeric(df['predicted_lecture'], errors='coerce') > 0]
    df['predicted_lecture'] = df['predicted_lecture'].astype(int)
    
    # 5. Group by Lecture
    lecture_groups = df.groupby('predicted_lecture')
    print(f"Found {len(lecture_groups)} lectures with videos.")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = None
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            print("OpenAI client initialized.")
        except Exception:
            pass

    results = {}
    
    # Iterate through each lecture group
    for lec_num, group in lecture_groups:
        lecture_key = f"Lecture {lec_num}"
        
        # Aggregate concepts and descriptions
        file_concepts_map = {}
        file_descriptions_map = {}
        
        # Use existing topic from metadata if available, though for YouTube we might not have a strong 'topic' column per file.
        # We'll rely on generating it.
        
        for _, row in group.iterrows():
            fname = row.get('file_name', 'unknown')
            # Parse sections for concepts
            sections_raw = row.get('sections', '')
            try:
                # Helper to parse key_concepts from sections JSON string
                # This matches logic from predict_lec_summary.py roughly
                parsed = _safe_json_or_literal_load(sections_raw)
                concepts = []
                if isinstance(parsed, list):
                    for sec in parsed:
                        if isinstance(sec, dict):
                            kc = sec.get('key_concept')
                            if isinstance(kc, str): concepts.append(kc)
                            elif isinstance(kc, list): concepts.extend([str(x) for x in kc])
                file_concepts_map[fname] = concepts[:10] # limit per file
            except:
                file_concepts_map[fname] = []
                
            # Description
            desc = str(row.get('description', '')).strip()
            if desc and desc.lower() != 'nan':
                 file_descriptions_map[fname] = [desc]
        
        out_topic = ''
        out_summary = ''

        if client is not None:
             # Construct prompt for OpenAI
            prompt = (
                f"Summarize CS 61A {lecture_key} based on the content of its video files.\n\n"
                "Per-file Key Concepts:\n"
                f"{json.dumps(file_concepts_map, ensure_ascii=False, indent=2)}\n\n"
                "Per-file Descriptions:\n"
                f"{json.dumps(file_descriptions_map, ensure_ascii=False, indent=2)}\n\n"
                "Rules:\n"
                "- topic: 3-8 words describing the lecture's main focus (e.g. 'Higher-Order Functions').\n"
                "- summary: 2-4 sentences; emphasize central ideas and conceptual progression.\n"
                "- Focus on the educational concepts.\n"
            )
            
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You summarize CS course lectures from video metadata."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=LectureSummary,
                    temperature=0.3,
                )
                parsed = completion.choices[0].message.parsed
                if parsed:
                    out_topic = parsed.topic.strip()
                    out_summary = parsed.summary.strip()
                    print(f"[+] {lecture_key}: {out_topic}")
            except Exception as e:
                print(f"[!] {lecture_key}: Error - {e}")
                out_topic = f"Lecture {lec_num}"
        else:
             out_topic = f"Lecture {lec_num}"
             out_summary = "Summary unavailable (no API key)"
             
        results[lecture_key] = {
            "topic": out_topic,
            "summary": out_summary
        }
    
    # Save output
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_path = os.path.join(output_dir, os.path.basename(out_json))
    
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(results)} summaries to: {final_output_path}")
    return results


if __name__ == "__main__":
    generate_openai_lecture_topics_json()
