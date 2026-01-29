# Quick Start Guide - New Slide-Based Pipeline

## What Changed?

### Old Approach ❌
- Filter calendar chunks from database
- Extract URLs and match to files
- Complex multi-step CSV processing

### New Approach ✅
- **Direct database query:** Filter slide files with `LIKE '%slides%'`
- **Single aggregation script:** Extract lecture numbers and group files
- **Simplified workflow:** 3 clean steps instead of 4+

## Run the Pipeline

### Step 1: Generate Lecture Summary from Slides
```bash
cd lec_slide_base_lec_summary
python generate_slide_summary_csv.py
```
**Output:** `output/cs_61a_lecture_summary.csv`

### Step 2: Generate OpenAI Topic Summaries
```bash
python lec_summary_json.py
```
**Output:** `output/cs_61a_lecture_topic_summaries.json`

### Step 3: Test Predictions (50 files)
```bash
python predict_lec_summary.py
```
**Output:** `output/cs_61a_video_lecture_eval.csv`

## Key Features

✅ **Database-Direct:** No calendar dependency  
✅ **Structured Output:** Pydantic models for reliable API calls  
✅ **Testing Mode:** Default `max_files=50` for cost efficiency  
✅ **Organized Outputs:** All results in `output/` folder  
✅ **Simplified CSV:** Only 3 columns in final output  

## Files to Use

### Active Scripts
- `generate_slide_summary_csv.py` - **NEW** Main aggregation script
- `lec_summary_json.py` - OpenAI summary generation
- `predict_lec_summary.py` - Lecture prediction & validation
- `filter_slide_files.py` - Optional debugging tool

### Deprecated (Can be removed)
- `filter_calendar_chunks.py` - Use generate_slide_summary_csv.py instead
- `lec_info_csv.py` - Functionality merged into generate_slide_summary_csv.py

## Requirements

```bash
pip install pandas openai python-dotenv pydantic
```

Create `.env`:
```
OPENAI_API_KEY=your_key_here
```

Database: `cs61a_metadata.db` in project root or parent directories

---
**Quick Reference:** Run all 3 scripts in order for complete pipeline
