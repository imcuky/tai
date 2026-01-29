# YouTube-Based Lecture Summary Pipeline

This pipeline generates lecture topic summaries from YouTube video transcripts and predicts which lecture each video belongs to. It follows a similar approach to the `lec_slide_base_lec_summary` pipeline but focuses on video content instead of slides.

## Overview

The pipeline consists of three main scripts that work together:

1. **filter_youtube_file.py** - Filters YouTube videos from the database
2. **youtube_lec_summary_json.py** - Generates lecture topic summaries using OpenAI
3. **eval_youtube_lec_summary.py** - Predicts which lecture each video belongs to

## Prerequisites

- Python 3.8+
- OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- Access to `cs61a_metadata.db` database
- Required Python packages:
  - pandas
  - openai
  - python-dotenv
  - pydantic

## Pipeline Workflow

### Step 1: Filter YouTube Files

**Script:** `filter_youtube_file.py`

Extracts all YouTube video records from the CS61A metadata database.

```bash
python filter_youtube_file.py
```

**What it does:**
- Connects to `cs61a_metadata.db`
- Queries for records where `relative_path` or `url` contains "youtube"
- Outputs: `output/cs61a_youtube_files.csv`

**Output columns include:**
- `file_name` - Name of the video file
- `relative_path` - Path to the file
- `url` - YouTube URL
- `sections` - Parsed transcript sections with key concepts
- `description` - Video description

### Step 2: Generate Lecture Topic Summaries

**Script:** `youtube_lec_summary_json.py`

Creates comprehensive topic summaries for each lecture using OpenAI's structured output API.

```bash
python youtube_lec_summary_json.py
```

**What it does:**
- Reads `cs_61a_lecture_summary.csv` (contains aggregated concepts per lecture)
- For each lecture, sends key concepts and descriptions to OpenAI
- Uses **structured output** (`beta.chat.completions.parse`) for reliable JSON responses
- Outputs: `output/cs_61a_youtube_lecture_topic_summaries.json`

**Input CSV columns expected:**
- `date` - Lecture date
- `topic` - Lecture topic hint
- `file_concepts_map` - JSON/dict mapping files to their key concepts
- `file_descriptions_map` - JSON/dict mapping files to their descriptions

**Output JSON structure:**
```json
{
  "Lecture 1": {
    "date": "Mon 8/26",
    "topic": "Functions and Control Flow",
    "summary": "Introduction to Python programming basics..."
  },
  "Lecture 2": {
    ...
  }
}
```

**OpenAI Parameters:**
- Model: `gpt-4o-mini` (default, can be changed)
- Temperature: 0.3 (for consistency)
- Max tokens: 300
- Response format: Pydantic `LectureSummary` model

### Step 3: Predict Lecture Assignments

**Script:** `eval_youtube_lec_summary.py`

Predicts which lecture each YouTube video file belongs to based on content matching.

```bash
python eval_youtube_lec_summary.py
```

**What it does:**
- Reads test files from `cs61a_test_files.csv` (YouTube videos to classify)
- Loads lecture summaries from `cs_61a_youtube_lecture_topic_summaries.json`
- For each video:
  - Extracts key concepts from `sections` field
  - Sends to OpenAI with lecture summaries for classification
  - Receives predicted lecture number using structured output
- Outputs: 
  - `output/cs61a_test_eval_prediction.csv` (full results with all columns)
  - `output/predictions_simple.csv` (simplified: relative_path, file_name, predicted_lecture only)

**Input files CSV columns:**
- `relative_path` - File path
- `file_name` - File name
- `description` - Video description
- `sections` - Parsed sections with key concepts

**Output CSV columns:**
- All original columns from input
- `predicted_lecture` - Predicted lecture number (1-40)
- `prediction_confidence` - Confidence level (low/medium/high)

**OpenAI Parameters:**
- Model: `gpt-4o-2024-08-06` (default)
- Temperature: 0.2 (for deterministic predictions)
- Response format: Pydantic `LecturePrediction` model

**Testing with Subset:**
```python
# Process only first 50 files for testing
python eval_youtube_lec_summary.py  # Default is 50 files

# To process all files, edit the script and set max_files=None
```

## Key Improvements Over Original Code

### 1. **Robust Path Resolution**
All scripts now intelligently search for input files in current directory and up to 3 parent directories:
```python
def resolve_path(file_path, max_levels=3):
    # Checks current dir, then ../file, ../../file, ../../../file
```

### 2. **Enhanced Error Handling**
- Detailed error messages with absolute paths
- Graceful fallbacks when API fails
- Safe Unicode handling for Windows terminals

### 3. **OpenAI Structured Output**
Using `beta.chat.completions.parse` with Pydantic models ensures:
- Reliable JSON responses (no parsing errors)
- Type-safe outputs
- Better error messages when model refuses

### 4. **Comprehensive Logging**
- Progress indicators ([+] success, [X] failure, [!] warning)
- Summary statistics at the end
- File counts and distribution analysis

### 5. **Better Documentation**
- Detailed docstrings for all functions
- Module-level documentation explaining purpose
- Type hints and parameter descriptions

## File Structure

```
youtube_base_lec_summary/
├── filter_youtube_file.py          # Step 1: Filter YouTube files
├── youtube_lec_summary_json.py     # Step 2: Generate summaries
├── eval_youtube_lec_summary.py     # Step 3: Predict lectures
├── filter_test_files.py            # Utility: Filter test files
├── README.md                        # This file
├── QUICKSTART.py                    # Quick reference guide
├── IMPROVEMENTS.md                  # Detailed improvements documentation
├── WORKFLOW.md                      # Visual workflow diagram
└── output/
    ├── cs61a_youtube_files.csv              # Filtered YouTube files
    ├── cs_61a_youtube_lecture_topic_summaries.json  # Lecture summaries
    ├── cs61a_test_eval_prediction.csv       # Full prediction results
    ├── predictions_simple.csv               # Simplified predictions
    └── cs61a_test_files.csv                 # Test files (if generated)
```

## Comparison with lec_slide_base_lec_summary

| Aspect | lec_slide_base_lec_summary | youtube_base_lec_summary |
|--------|---------------------------|-------------------------|
| **Data Source** | Lecture slides, discussions, tutorials | YouTube video transcripts |
| **Filter Target** | `%slide%`, `%disc%`, `%tutorial%` | `%youtube%` in path/url |
| **Key Concepts** | From slide content sections | From video transcript sections |
| **Summary Input** | Slide text and structure | Video transcripts and descriptions |
| **Prediction Focus** | Matching slides to lectures | Matching videos to lectures |

## Usage Examples

### Full Pipeline Execution

```bash
# Step 1: Extract YouTube files
python filter_youtube_file.py

# Step 2: Generate lecture summaries (requires OPENAI_API_KEY)
python youtube_lec_summary_json.py

# Step 3: Predict lecture assignments
python eval_youtube_lec_summary.py
```

### Custom Parameters

```python
# Custom database path
from filter_youtube_file import filter_youtube_files
filter_youtube_files(db_path="../data/custom.db", output_csv="youtube_vids.csv")

# Custom model for summary generation
from youtube_lec_summary_json import generate_openai_lecture_topics_json
generate_openai_lecture_topics_json(
    summary_csv="my_summary.csv",
    out_json="my_topics.json",
    model="gpt-4o"  # Use more powerful model
)

# Custom prediction model with file limit for testing
from eval_youtube_lec_summary import predict_lectures
predict_lectures(
    files_csv="my_files.csv",
    topics_json="my_topics.json",
    model="gpt-4o-2024-08-06",
    max_files=50  # Process only first 50 files for testing
)

# Process all files (production)
predict_lectures(max_files=None)  # None = process all files
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

## Expected Costs

Approximate OpenAI API costs (as of 2024):

- **Summary generation** (Step 2):
  - ~40 lectures × 300 tokens output = 12K tokens
  - Input: ~40 lectures × 500 tokens = 20K tokens
  - Total: ~$0.01 - $0.05

- **Prediction** (Step 3):
  - ~200 files × 200 tokens = 40K tokens
  - Input: ~200 files × 1000 tokens = 200K tokens
  - Total: ~$0.10 - $0.50

**Total estimated cost per run: $0.15 - $0.60**

## Troubleshooting

### Error: "Database file not found"
- Ensure `cs61a_metadata.db` is in the workspace root or specify full path
- Check file permissions

### Error: "OPENAI_API_KEY not found"
- Create `.env` file with your API key
- Ensure `.env` is in the project root or parent directories

### Error: "Input file not found"
- Run scripts in order (Step 1 → Step 2 → Step 3)
- Check that previous steps completed successfully

### Unicode Errors on Windows
- Already handled with safe encoding/decoding
- If issues persist, run in PowerShell with UTF-8: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`

## Contributing

When modifying the code:
1. Maintain the structured output format (Pydantic models)
2. Add comprehensive error handling
3. Update docstrings and this README
4. Test with sample data before production runs

## References

- OpenAI Structured Output: https://platform.openai.com/docs/guides/structured-outputs
- Pydantic Models: https://docs.pydantic.dev/
- Similar pipeline: `../lec_slide_base_lec_summary/`
