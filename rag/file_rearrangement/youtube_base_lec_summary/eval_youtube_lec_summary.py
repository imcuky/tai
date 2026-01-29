import pandas as pd
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import ast
import re


class LecturePrediction(BaseModel):
    """Structured output model for lecture predictions."""
    lecture_number: int
    # confidence: str = "medium"  # Optional: low, medium, high


def _safe_json_or_literal_load(s):
    """
    Parse a JSON or Python literal string safely.
    
    Args:
        s: String to parse
    
    Returns:
        Parsed object or None on failure
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
            return ast.literal_eval(st)
        except Exception:
            return None



def derive_gt_from_file_path(file_path: str) -> tuple[int | None, str | None]:
    """Derive numeric lecture and category label from a canonical file_path like
    'CS 61A/study/lecture/lec03/youtube03/Control/1-Multiple Environments.webm'.
    Returns (lecture_number, category_label) where lecture_number is int or None
    and category_label is a string (e.g., 'Control') or None.
    """
    if not file_path or not isinstance(file_path, str):
        return None, None
    parts = re.split(r"[\\/]+", file_path)
    parts_clean = [p for p in parts if p]
    # lecture number
    lec_num = None
    for p in parts_clean:
        m = re.match(r"lec(\d+)", p, flags=re.IGNORECASE)
        if m:
            try:
                lec_num = int(m.group(1))
                break
            except Exception:
                pass
    # category label: segment after youtubeXX
    cat_label = None
    for i, p in enumerate(parts_clean):
        if re.match(r"youtube\d+", p, flags=re.IGNORECASE):
            if i + 1 < len(parts_clean):
                cat_label = parts_clean[i + 1]
            break
    return lec_num, cat_label


def _normalize_label(s: str) -> set[str]:
    if not s:
        return set()
    t = s.lower()
    # Remove term/session annotations like (Su25), (Fa24), years
    t = re.sub(r"\([^\)]*\)", " ", t)
    t = re.sub(r"\b(su|fa|sp|wi)\d{2}\b", " ", t)
    t = re.sub(r"\d{4}", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    tokens = [w for w in t.split() if w and w not in {"cs", "61a", "lecture", "and", "the", "of", "in", "for"}]
    return set(tokens)


def _category_match(gt_label: str, topic: str) -> bool:
    gt_tokens = _normalize_label(gt_label)
    topic_tokens = _normalize_label(topic)
    if not gt_tokens or not topic_tokens:
        return False
    # Consider match if there is at least one meaningful token overlap
    return len(gt_tokens & topic_tokens) > 0


def _topic_for_lecture(lectures_data, number: int) -> str:
    for lec in lectures_data:
        if lec.get('lecture_number') == number:
            return lec.get('topic_generated') or lec.get('topic') or ""
    return ""


def parse_key_concepts(sections_str):
    """
    Parse the 'sections' string (JSON or Python literal) to extract key concepts.
    
    This extracts key_concept fields from the sections data structure, which
    contains the parsed semantic information about video content.
    
    Args:
        sections_str: JSON/literal string containing sections data
    
    Returns:
        list: List of unique key concepts
    """
    if not isinstance(sections_str, str): 
        return []
    
    data = _safe_json_or_literal_load(sections_str)
    if not data:
        return []
    
    concepts = []
    seen = set()
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Extract key_concept field
                kc = item.get('key_concept')
                if kc:
                    vals = []
                    if isinstance(kc, list): 
                        vals = kc
                    elif isinstance(kc, str): 
                        vals = [kc]
                    
                    for v in vals:
                        if isinstance(v, str) and v.strip():
                            concept = v.strip()
                            if concept.lower() not in seen:
                                seen.add(concept.lower())
                                concepts.append(concept)
    
    return concepts


def resolve_path(file_path, max_levels=3):
    """
    Resolve file path by checking current and parent directories.
    
    Args:
        file_path: Path to resolve
        max_levels: Maximum number of parent directories to check
    
    Returns:
        str: Resolved path, or original path if not found
    """
    if os.path.exists(file_path):
        return file_path
    
    # Search parent directories
    for levels_up in range(1, max_levels + 1):
        potential_path = os.path.join("../" * levels_up, file_path)
        if os.path.exists(potential_path):
            return potential_path
    
    return file_path


def predict_lectures(
    files_csv="cs61a_test_files.csv", 
    topics_json="cs_61a_youtube_lecture_topic_summaries.json",
    output_csv="cs61a_test_eval_prediction.csv",
    model="gpt-4o-2024-08-06",
    max_files=None
):
    """
    Predict which lecture each YouTube video file belongs to.
    
    Uses the lecture topic summaries JSON and OpenAI structured output to
    classify each video file to its most likely lecture.
    
    Args:
        files_csv: Path to CSV containing YouTube files to classify
        topics_json: Path to JSON with lecture topic summaries
        output_csv: Output CSV path for predictions
        model: OpenAI model to use
        max_files: Maximum number of files to process (None for all). Use for testing.
    
    Returns:
        pd.DataFrame: DataFrame with predictions, or None on error
    """
    # Resolve input file paths
    real_files_csv = resolve_path(files_csv)
    real_topics_json = resolve_path(topics_json)

    # Validate input files exist
    if not os.path.exists(real_files_csv):
        print(f"Error: Input file not found: {files_csv}")
        print(f"Searched: {os.path.abspath(real_files_csv)}")
        return None
    
    if not os.path.exists(real_topics_json):
        print(f"Error: Input file not found: {topics_json}")
        print(f"Searched: {os.path.abspath(real_topics_json)}")
        return None

    print(f"Loading files from:")
    print(f"  Files CSV: {real_files_csv}")
    print(f"  Topics JSON: {real_topics_json}")
    
    # Load data
    df_files = pd.read_csv(real_files_csv)
    
    # Limit to subset for testing if specified
    if max_files is not None and max_files > 0:
        original_count = len(df_files)
        df_files = df_files.head(max_files)
        print(f"\nLimited to first {len(df_files)} files out of {original_count} for testing.")
    
    try:
        with open(real_topics_json, 'r', encoding='utf-8') as f:
            lectures_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

    print(f"\nLoaded {len(df_files)} files and {len(lectures_data)} lectures.")

    # Format lectures for prompt
    sorted_lecs = []
    for k, v in lectures_data.items():
        # k is "Lecture 1", "Lecture 2", etc.
        try:
            m = re.search(r"(\d+)", k)
            if m:
                num = int(m.group(1))
                sorted_lecs.append((num, k, v))
        except:
            continue
    
    sorted_lecs.sort(key=lambda x: x[0])

    lectures_text_list = []
    for num, k, v in sorted_lecs:
        summary = v.get('summary', '')
        topic = v.get('topic', '')
        lectures_text_list.append(f"{num}. {topic}. {summary}")

    lectures_block = "\n".join(lectures_text_list)
    
    print(f"\nLecture summaries loaded:")
    print("=" * 60)
    print(lectures_block[:500] + "..." if len(lectures_block) > 500 else lectures_block)
    print("=" * 60)

    # Initialize OpenAI client
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY not found in environment.")
        print("Cannot proceed with prediction.")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        print("\nOpenAI client initialized successfully.")
    except Exception as e:
        print(f"\nError initializing OpenAI client: {e}")
        return None

    # Predict lecture for each file
    predictions = []
    confidences = []
    
    print(f"\nStarting predictions for {len(df_files)} files...")
    print("=" * 60)

    for idx, row in df_files.iterrows():
        file_path = row.get('relative_path', '')
        file_name = row.get('file_name', '')
        desc = row.get('description', '') if 'description' in row else ""
        sections = row.get('sections', '')
        
        # Parse key concepts from sections
        key_concepts = parse_key_concepts(sections)
        
        # Construct file information
        file_info = (
            f"File: {file_name}\n"
            f"Path: {file_path}\n"
            f"Description: {desc}\n"
            f"Key Concepts: {', '.join(key_concepts[:15])}"  # Limit to avoid token overflow
        )
        
        # Create prompt
        sys_prompt = (
            "You are an expert at classifying CS study materials into course lecture categories.\n\n"
            "Here is the list of Lectures in the course:\n"
            f"{lectures_block}\n\n"
            "Task: Givn the materials, Which lecture number does this file most likely belong to based on topic match?\n"
            "Consider the key concepts, description, and file path when making your decision.\n"
            "Return the lecture number as an integer. If unsure, pick the best match."
        )
        prompt = (

            "Here is the file to classify:\n"
            f"{file_info}\n\n"
        )

        pred_lec = -1
        # confidence = "low"
        
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": sys_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format=LecturePrediction,
                temperature=0.2,
            )
            
            parsed_resp = completion.choices[0].message.parsed
            if parsed_resp:
                pred_lec = parsed_resp.lecture_number
                # confidence = getattr(parsed_resp, 'confidence', 'medium')
            else:
                print(f"  [!] Parsing failed for: {file_name}")
                
        except Exception as e:
            print(f"  [X] Error for {file_name}: {e}")
            pred_lec = -1

        # Safe print to avoid Unicode errors on Windows - use ASCII-safe symbols
        safe_name = file_name.encode('ascii', errors='replace').decode('ascii')
        status = "[+]" if pred_lec > 0 else "[X]"
        print(f"  {status} [{idx+1}/{len(df_files)}] {safe_name[:50]:50s} -> Lecture {pred_lec:2d}")
        
        predictions.append(pred_lec)
        # confidences.append(confidence)

    # Add predictions to dataframe
    df_files['predicted_lecture'] = predictions
    # df_files['prediction_confidence'] = confidences
    
    # Create output directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save simplified output with only requested columns
    df_simplified = df_files[['relative_path', 'file_name', 'predicted_lecture']].copy()
    df_simplified.to_csv(output_csv, index=False)
    
    # Calculate and print accuracy stats
    print_accuracy_stats(df_files, lectures_data)
    
    print(f"\nPredictions saved to: {output_csv}")
    print(f"Absolute path: {os.path.abspath(output_csv)}")
    
    return df_files


def print_accuracy_stats(df_files, lectures_data):
    """
    Calculate and display accuracy statistics for lecture predictions.
    
    Attempts to derive ground truth from file paths or names to verify predictions.
    This logic mirrors the evaluation in predict_lec_summary.py using category-aware matching.
    """
    total = len(df_files)
    correct_count = 0
    ground_truth_count = 0
    
    if total == 0:
        return

    print("\n" + "="*30)
    print("SUMMARY STATISTICS")
    print("="*30)
    
    for _, row in df_files.iterrows():
        path = str(row.get('relative_path', ''))
        pred = row.get('predicted_lecture', -1)
        if hasattr(pred, 'isdigit') and not pred.isdigit():
             pred = -1 # Handle non-numeric or empty
        try:
             pred = int(float(pred)) if pred else -1
        except:
             pred = -1
        
        # Try to find ground truth lecture number
        gt_lec, gt_cat = derive_gt_from_file_path(path)
            
        if gt_lec is not None:
            is_correct = (gt_lec == pred)
            
            # If simplistic numeric match failed, try category match
            if not is_correct and gt_cat:
                pred_topic = _topic_for_lecture(lectures_data, pred)
                if _category_match(gt_cat, pred_topic):
                    is_correct = True
            
            ground_truth_count += 1
            if is_correct:
                correct_count += 1
    
    print(f"Total files processed: {total}")
    print(f"Files with identifiable ground truth: {ground_truth_count}")
    
    if ground_truth_count > 0:
        accuracy = (correct_count / ground_truth_count) * 100
        print(f"OpenAI acc (category-aware): {correct_count}/{ground_truth_count} ({accuracy:.1f}%)")
    else:
        print("No ground truth labels found in file paths/names for accuracy calculation.")



if __name__ == "__main__":
    # For testing: limit to 50 files. Remove max_files parameter or set to None for all files.
    result = predict_lectures(max_files=20)
    if result is not None:
        print(f"\nSuccess! Processed {len(result)} files.")
