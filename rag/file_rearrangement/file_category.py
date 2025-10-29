import pandas as pd
import json
import os
from dotenv import load_dotenv
import time
import argparse
from typing import List, Dict, Optional
from openai import OpenAI

def load_cs61a_files(csv_path: str) -> pd.DataFrame:
    """Load CS 61A files CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} files from {csv_path}")
        
        # Show sample data
        print("\nSample file data:")
        for i, row in df.head(3).iterrows():
            print(f"  - {row.get('file_name', 'N/A')}")
            print(f"    Path: {row.get('relative_path', 'N/A')}")
            print(f"    URL: {row.get('url', 'N/A')}")
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def build_categorization_prompt(file_name: str, relative_path: str, url: str) -> str:
    """Build prompt for file categorization"""
    
    prompt = f"""
You are categorizing CS 61A course files into exactly one of these categories:

**Categories:**
- "Lecture": Lecture notes, slides, recordings, lecture materials
- "Practice": Homework, labs, projects, practice problems, assignments, exams, quizzes
- "Support": Administrative files, syllabus, course info, resources, guides, references

**File Information:**
- File name: {file_name or 'N/A'}
- Relative path: {relative_path or 'N/A'}
- URL: {url or 'N/A'}

**Instructions:**
- Return ONLY one word: "Lecture", "Practice", or "Support"
- No explanation needed
- Base decision on file name, path, and URL patterns
- If unclear, use best judgment based on typical CS course structure

**Examples:**
- "lec01.pdf" → Lecture
- "hw01.py" → Practice
- "syllabus.html" → Support
- "lab02/" → Practice
- "disc03.pdf" → Practice
- "calendar/" → Support

Category:"""
    
    return prompt.strip()

def categorize_with_openai(file_name: str, relative_path: str, url: str, 
                          client: OpenAI, model: str = "gpt-4o-mini", 
                          max_retries: int = 3) -> str:
    """Categorize a single file using OpenAI API"""
    
    prompt = build_categorization_prompt(file_name, relative_path, url)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise file categorizer. Return only one word: Lecture, Practice, or Support."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            category = response.choices[0].message.content.strip()
            
            # Validate response
            valid_categories = ["Lecture", "Practice", "Support"]
            if category in valid_categories:
                return category
            
            # Try to extract valid category from response
            for valid_cat in valid_categories:
                if valid_cat.lower() in category.lower():
                    return valid_cat
            
            # Default fallback
            return "Support"
            
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "Support"  # Default fallback
    
    return "Support"

def categorize_files_batch(df: pd.DataFrame, client: OpenAI, model: str = "gpt-4o-mini") -> pd.DataFrame:
    """Categorize all files in the dataframe"""
    
    categories = []
    total_files = len(df)
    
    print(f"\nCategorizing {total_files} files...")
    
    for i, row in df.iterrows():
        file_name = str(row.get('file_name', ''))
        relative_path = str(row.get('relative_path', ''))
        url = str(row.get('url', ''))
        uuid = row.get('uuid', '')
        
        # Show progress
        if i % 10 == 0:
            print(f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
        
        # Categorize file
        category = categorize_with_openai(file_name, relative_path, url, client, model)
        categories.append(category)
        
        # Rate limiting - be nice to API
        time.sleep(0.5)
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'uuid': df['uuid'],
        'file_name': df['file_name'],
        'category': categories
    })
    
    return result_df

def heuristic_categorization(file_name: str, relative_path: str, url: str) -> str:
    """Fallback heuristic categorization if no API key"""
    
    text = f"{file_name} {relative_path} {url}".lower()
    
    # Lecture patterns
    lecture_patterns = [
        'lec', 'lecture', 'slides', 'notes', 'recording', 'video',
        'slide', 'presentation', 'ppt'
    ]
    
    # Practice patterns  
    practice_patterns = [
        'hw', 'homework', 'lab', 'project', 'proj', 'disc', 'discussion',
        'exam', 'quiz', 'test', 'midterm', 'final', 'assignment', 'prob',
        'problem', 'exercise', 'practice', 'sol', 'solution'
    ]
    
    # Support patterns
    support_patterns = [
        'syllabus', 'calendar', 'admin', 'info', 'guide', 'resource',
        'reference', 'help', 'faq', 'policy', 'schedule', 'announcement',
        'piazza', 'staff', 'office', 'hour'
    ]
    
    # Check patterns
    for pattern in lecture_patterns:
        if pattern in text:
            return "Lecture"
    
    for pattern in practice_patterns:
        if pattern in text:
            return "Practice"
    
    for pattern in support_patterns:
        if pattern in text:
            return "Support"
    
    # Default to Support if unclear
    return "Support"

def categorize_files_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize files using heuristic rules (no API)"""
    
    categories = []
    
    print(f"\nCategorizing {len(df)} files using heuristic rules...")
    
    for i, row in df.iterrows():
        file_name = str(row.get('file_name', ''))
        relative_path = str(row.get('relative_path', ''))
        url = str(row.get('url', ''))
        
        category = heuristic_categorization(file_name, relative_path, url)
        categories.append(category)
    
    result_df = pd.DataFrame({
        'uuid': df['uuid'],
        'file_name': df['file_name'], 
        'category': categories
    })
    
    return result_df

def analyze_categorization_results(result_df: pd.DataFrame):
    """Analyze and display categorization results"""
    
    print(f"\nCategorization Results:")
    print("=" * 40)
    
    category_counts = result_df['category'].value_counts()
    total = len(result_df)
    
    for category, count in category_counts.items():
        percentage = count / total * 100
        print(f"{category}: {count} files ({percentage:.1f}%)")
    
    print(f"\nTotal files categorized: {total}")
    
    # Show sample files per category
    print(f"\nSample files per category:")
    for category in ["Lecture", "Practice", "Support"]:
        print(f"\n{category}:")
        sample_files = result_df[result_df['category'] == category]['file_name'].head(5)
        for i, file_name in enumerate(sample_files, 1):
            print(f"  {i}. {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Categorize CS 61A files into Lecture/Practice/Support")
    parser.add_argument("--input", "-i", default="cs61a_files.csv", help="Input CSV file path")
    parser.add_argument("--output", "-o", default="cs61a_file_categories.csv", help="Output CSV file path_with_API")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--no-api", action="store_true", help="Use heuristic rules instead of OpenAI API")
    args = parser.parse_args()
    
    print("CS 61A File Categorization")
    print("=" * 30)
    

    df = load_cs61a_files(args.input)
    if df is None:
        return
    
    # Initialize OpenAI client if API key available
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    use_api = not args.no_api and api_key is not None
    
    if use_api:
        print(f"\nUsing OpenAI API with model: {args.model}")
        client = OpenAI(api_key=api_key)
        result_df = categorize_files_batch(df, client, args.model)
    else:
        if not api_key:
            print(f"\nWARNING: No OPENAI_API_KEY found, using heuristic rules")
        else:
            print(f"\nUsing heuristic rules (--no-api specified)")
        result_df = categorize_files_heuristic(df)
    
    # Analyze results
    analyze_categorization_results(result_df)
    
    # Save results
    result_df.to_csv(args.output, index=False)
    print(f"\nSUCCESS: Results saved to: {args.output}")
    
    print(f"\nOutput format:")
    print("uuid,file_name,category")
    print(result_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()