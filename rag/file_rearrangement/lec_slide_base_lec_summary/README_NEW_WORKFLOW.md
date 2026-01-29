# Slide-Based Lecture Summary Pipeline - Updated Workflow

## Overview
This pipeline has been **simplified** to work directly with the updated cs61a_metadata.db. The old calendar-based filtering approach has been replaced with direct database queries.

## New Simplified Workflow

### Old Workflow (Deprecated)
```
filter_calendar_chunks.py → lec_info_csv.py → lec_summary_json.py → predict_lec_summary.py
```
❌ Complex, multi-step process
❌ Relied on calendar chunks
❌ Multiple intermediate CSV files

### New Workflow (Current)
```
generate_slide_summary_csv.py → lec_summary_json.py → predict_lec_summary.py
```
✅ Simplified, direct database access
✅ Fewer intermediate files
✅ Consistent with youtube_base_lec_summary pattern

## Pipeline Steps

### Step 1: Generate Slide Lecture Summary CSV
**Script:** `generate_slide_summary_csv.py`

**What it does:**
1. Connects to cs61a_metadata.db
2. Filters all slide files (relative_path contains "slides")
3. Extracts lecture numbers from paths (e.g., "slides01" → Lecture 1)
4. Parses key concepts from sections field
5. Aggregates files by lecture number
6. Creates per-lecture summary with:
   - Key concepts list
   - File-to-concepts mapping
   - File-to-descriptions mapping
   - Slide file count

**Output:** `output/cs_61a_lecture_summary.csv`

**Run:**
```bash
python generate_slide_summary_csv.py
```

**Expected Output:**
```
Connecting to database: cs61a_metadata.db
Found 150 slide file records.
Filtered to 145 files with valid lecture numbers.

Generated summaries for 38 lectures.
Lecture range: 1 to 38

Sample lecture summaries:
  Lecture 1: 4 files, 12 key concepts
  Lecture 2: 3 files, 8 key concepts
  Lecture 3: 5 files, 15 key concepts
```

### Step 2: Generate Lecture Topic Summaries (OpenAI)
**Script:** `lec_summary_json.py`

**What it does:**
1. Reads `output/cs_61a_lecture_summary.csv`
2. Uses OpenAI structured output to generate:
   - Concise topic (3-8 words)
   - Comprehensive summary (2-4 sentences)
3. Saves to JSON for downstream use

**Output:** `output/cs_61a_lecture_topic_summaries.json`

**Run:**
```bash
python lec_summary_json.py
```

**Expected Output:**
```
[+] Lecture 1: Functions and Expressions
[+] Lecture 2: Names, Assignment, and Environment
[+] Lecture 3: Control and Iteration
...
Generated 38 lecture summaries.
```

### Step 3: Predict Lecture Classifications (Testing Mode)
**Script:** `predict_lec_summary.py`

**What it does:**
1. Loads lecture summaries JSON
2. Loads files from database
3. Uses OpenAI to predict which lecture each file belongs to
4. Validates predictions against ground truth

**Output:** `output/cs_61a_video_lecture_eval.csv`

**Run:**
```bash
python predict_lec_summary.py
```

**Testing Mode (50 files):**
The script defaults to `max_files=50` for cost-effective testing.

## Key Improvements

### 1. Database-Direct Filtering
✅ **Before:** Calendar chunks → extract URLs → match files  
✅ **After:** Direct query on file table with LIKE '%slides%'

### 2. Simplified Aggregation
✅ **Before:** Multiple CSV processing steps  
✅ **After:** Single script aggregates by lecture_number

### 3. Structured Output (OpenAI)
✅ Uses Pydantic models for reliable JSON responses  
✅ No more manual JSON parsing or regex extraction  
✅ Consistent with youtube_base_lec_summary implementation

### 4. Organized Outputs
✅ All outputs in `output/` subfolder  
✅ Simplified CSV with only essential columns  
✅ Progress indicators with ASCII-safe symbols

## File Descriptions

### Active Scripts
- **generate_slide_summary_csv.py** - NEW: Direct DB filtering and aggregation
- **lec_summary_json.py** - UPDATED: Uses structured output API
- **predict_lec_summary.py** - UPDATED: Uses structured output API
- **filter_slide_files.py** - OPTIONAL: Standalone slide file filter (for debugging)

### Deprecated Scripts
- ~~filter_calendar_chunks.py~~ - Replaced by generate_slide_summary_csv.py
- ~~lec_info_csv.py~~ - Functionality merged into generate_slide_summary_csv.py
- ~~file_category.py~~ - Category info now in database

## Database Schema

### Required Fields from `file` table:
```sql
SELECT file_name, relative_path, file_path, sections, description, category
FROM file 
WHERE lower(relative_path) LIKE '%slides%'
```

### Lecture Number Extraction:
- Pattern: `slides01`, `slides1`, `lec01`, `lec1`
- Regex: `(?:slides|lec)0*(\d+)`
- Example: `slides01/intro.pdf` → Lecture 1

## Usage Examples

### Quick Start (Testing Mode)
```bash
# Generate lecture summary CSV from slides
python generate_slide_summary_csv.py

# Generate OpenAI-powered topic summaries
python lec_summary_json.py

# Test predictions on 50 files
python predict_lec_summary.py
```

### Full Production Run
Edit `predict_lec_summary.py` and set `max_files=None`:
```python
def main(max_files=None):  # Changed from max_files=50
    # ... rest of function
```

Then run:
```bash
python predict_lec_summary.py
```

## Output Files

All outputs are saved to `output/` subfolder:

1. **cs_61a_lecture_summary.csv**
   - Lecture number, key concepts, file mappings
   - Generated by: generate_slide_summary_csv.py

2. **cs_61a_lecture_topic_summaries.json**
   - Lecture topics and summaries from OpenAI
   - Generated by: lec_summary_json.py

3. **cs_61a_video_lecture_eval.csv**
   - Simplified: relative_path, file_name, predicted_lecture
   - Generated by: predict_lec_summary.py

## Environment Requirements

### Python Packages
```bash
pip install pandas openai python-dotenv pydantic
```

### Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### Database
- **File:** cs61a_metadata.db
- **Table:** file
- **Required columns:** file_name, relative_path, sections, description

## Troubleshooting

### No slide files found
- Check database path resolution
- Verify slide files exist with `SELECT * FROM file WHERE lower(relative_path) LIKE '%slides%'`
- Check file_path vs relative_path fields

### Unicode errors
- All progress indicators use ASCII-safe symbols: [+] [X] [!]
- CSV outputs use UTF-8 encoding

### OpenAI API errors
- Verify OPENAI_API_KEY in .env
- Check API quota and rate limits
- Structured output requires compatible models (gpt-4o-mini, gpt-4o)

## Migration Notes

### For existing users:
1. ✅ Old CSV outputs are compatible
2. ✅ JSON structure unchanged
3. ✅ Can run new scripts alongside old ones
4. ⚠️ Recommend switching to new workflow for simplicity

### Breaking changes:
- ❌ filter_calendar_chunks.py no longer used
- ❌ lec_info_csv.py functionality replaced
- ✅ All outputs remain backward compatible

---
**Updated:** January 28, 2026  
**Pipeline Version:** 2.0 (Database-Direct)  
**Status:** Production Ready
