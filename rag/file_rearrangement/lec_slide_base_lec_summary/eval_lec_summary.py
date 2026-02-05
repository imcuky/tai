"""
Evaluation script for Lecture Summary Classification.
Uses generated lecture topic summaries (JSON) to classify YouTube videos into lecture numbers.
Evaluates performance by matching predicted lecture number against ground truth derived from file paths.
"""

import os
import json
import sqlite3
import pandas as pd
import re
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Data Models and Helpers
# -----------------------------------------------------------------------------

class LecturePrediction(BaseModel):
    """Structured output model for lecture predictions."""
    lecture_number: int
    reason: str = ""

# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

def load_lecture_topics(json_path="output/cs_61a_lecture_topic_summaries.json"):
    """Load the generated lecture topic summaries."""
    # Handle relative path from script execution
    if not os.path.exists(json_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "output", "cs_61a_lecture_topic_summaries.json")
    
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lectures = []
        for key, val in data.items():
            # Extract number from "Lecture 1"
            m = re.search(r'\d+', key)
            if m:
                # Flexible key handling for topic/summary
                topic = val.get("lecture_topic") or val.get("topic") or ""
                summary = val.get("lecture_summary") or val.get("summary") or ""
                
                lectures.append({
                    "lecture_number": int(m.group(0)),
                    "topic": topic,
                    "summary": summary
                })
        return sorted(lectures, key=lambda x: x['lecture_number'])
    except Exception as e:
        print(f"Error loading topics JSON: {e}")
        return []

def load_test_videos(db_path="cs61a_metadata.db", seed=42):
    """Load YouTube video metadata from database and shuffle."""
    if not os.path.exists(db_path):
        # Try looking up directory tree
        if os.path.exists(os.path.join("..", db_path)):
            db_path = os.path.join("..", db_path)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        # Filter for videos with relevant metadata
        # We specifically look for files with 'youtube' in the relative_path
        query = """
            SELECT file_name, relative_path, url, description 
            FROM file 
            WHERE lower(relative_path) LIKE '%youtube%' 
        """
        df = pd.read_sql_query(query, conn)
        
        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return df
    finally:
        conn.close()

def derive_gt_from_path(file_path):
    """Extract lecture number from file path like '.../lec03/youtube03/...'"""
    if not isinstance(file_path, str):
        return None
    # Look for 'lec' followed by digits, case insensitive
    m = re.search(r'lec0*(\d+)', file_path, re.IGNORECASE)
    return int(m.group(1)) if m else None

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main(limit=None):
    print("="*60)
    print("Lecture Summary Classification Eval (Descript       ion Only)")
    print("="*60)

    # 1. Setup OpenAI
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return
    client = OpenAI(api_key=api_key)

    # 2. Load Data
    lectures = load_lecture_topics()
    if not lectures:
        print("No lecture topics found. Run generate_slide_summary_csv.py first.")
        return 
    print(f"Loaded {len(lectures)} lecture topics.")

    # Pre-compute lectures string once
    lectures_str = json.dumps([
        {
            "num": l["lecture_number"], 
            "topic": l["topic"], 
            "summary": l["summary"][:200] + "..." if len(l["summary"]) > 200 else l["summary"]
        } 
        for l in lectures
    ], ensure_ascii=False)

    def classify_video(video_meta, model="gpt-4o"):
        """Inner function to classify video using captured context."""
        # Use ONLY description as requested
        description = video_meta.get("description", "")
        if not description or pd.isna(description):
            description = "No description available."
        else:
            description = str(description)[:1000] # truncate if too long

        video_str = json.dumps({
            "description": description
        }, ensure_ascii=False)

        prompt = (
            f"Video Info:\n{video_str}\n\n"
            "Return the single most appropriate lecture number."
        )

        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at classifying CS course materials.\n"
                    "Identify which Lecture Number the following video belongs to based on its description.\n"
                    f"Available Lectures:\n{lectures_str}\n\n"
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=LecturePrediction,
                temperature=0.0
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    # Load and shuffle videos
    videos_df = load_test_videos(seed=42)
    if videos_df.empty:
        print("No videos found in database.")
        return
    
    print(f"Loaded {len(videos_df)} videos (shuffled).")

    # 3. Predict & Evaluate
    results = []
    correct_count = 0
    total_eval = 0

    # Limit for testing if needed, or iterate all
    if limit:
        testing_subset = videos_df.head(limit)
        print(f"Limiting evaluation to first {limit} samples.")
    else:
        testing_subset = videos_df 

    print(f"\nEvaluate {len(testing_subset)} samples...")


    for idx, row in testing_subset.iterrows():
        # Determine Ground Truth from path
        gt_lec = derive_gt_from_path(row['relative_path'])
        
        if gt_lec is None:
            # print(f"Skipping {row['file_name']}: Could not derive lecture number from path.")
            continue 

        # Classify
        pred = classify_video(row)
        pred_lec = pred.lecture_number if pred else -1
        reason = pred.reason if pred else "Error"

        # Check correctness
        is_correct = (pred_lec == gt_lec)
        
        if is_correct:
            correct_count += 1
        total_eval += 1
        
        status = "OK" if is_correct else "XM"
        # Ensure name is printable in console
        safe_name = str(row['file_name']).encode('ascii', 'ignore').decode('ascii')[:30]
        print(f"[{status}] {safe_name:<30} | GT: {gt_lec:<2} | Pred: {pred_lec:<2}")

        results.append({
            "file_name": row['file_name'],
            "relative_path": row['relative_path'],
            "ground_truth": gt_lec,
            "prediction": pred_lec,
            "correct": is_correct,
            "reason": reason
        })

    # 4. Save and Summarize
    if total_eval > 0:
        accuracy = (correct_count / total_eval) * 100
        print(f"\nFinal Accuracy: {correct_count}/{total_eval} ({accuracy:.2f}%)")
        
        out_file = os.path.join(os.path.dirname(__file__), "output", "eval_summary_results.csv")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"Saved results to {out_file}")
    else:
        print("No valid evaluation samples found.")

if __name__ == "__main__":
    main(limit=20) # Uncomment to limit to 20 samples
