# OpenAI Structured Output Update

## Overview
Updated `lec_slide_base_lec_summary` codebase to use OpenAI's structured output API with Pydantic models, matching the implementation in `youtube_base_lec_summary`.

## Changes Made

### 1. lec_summary_json.py

**Added:**
- `from pydantic import BaseModel` import
- `LectureSummary` Pydantic model for structured output:
  ```python
  class LectureSummary(BaseModel):
      """Structured output model for lecture summaries."""
      topic: str
      summary: str
  ```

**Updated:**
- Replaced manual JSON parsing with `client.beta.chat.completions.parse()`
- Uses `response_format=LectureSummary` for reliable structured responses
- Simplified error handling (no more regex extraction, code fence stripping, etc.)
- Added progress indicators with ASCII-safe symbols: `[+]`, `[!]`, `[X]`

**Benefits:**
- No more JSON parsing errors or malformed responses
- Guaranteed schema compliance
- Cleaner, more maintainable code
- Consistent with youtube_base_lec_summary implementation

### 2. predict_lec_summary.py

**Added:**
- `from pydantic import BaseModel` import
- `LecturePrediction` Pydantic model:
  ```python
  class LecturePrediction(BaseModel):
      """Structured output model for lecture predictions."""
      lecture_number: int
      confidence: str = "medium"  # Optional: low, medium, high
      reason: str = ""  # Optional: explanation for the prediction
  ```

**Updated:**
- `call_openai_classify()` function now uses `beta.chat.completions.parse()`
- Uses `response_format=LecturePrediction` for structured output
- Added confidence mapping: `{"low": 0.3, "medium": 0.6, "high": 0.9}`
- Improved error handling with structured response parsing
- Extracts `reason` field for debugging/transparency

**Benefits:**
- Reliable integer lecture number predictions
- Structured confidence levels
- Optional reasoning for predictions (useful for debugging)
- Consistent API usage pattern

## Technical Details

### Structured Output API
```python
completion = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "System prompt..."},
        {"role": "user", "content": "User prompt..."}
    ],
    response_format=PydanticModel,
    temperature=0.3,
    max_tokens=300,
)

parsed_response = completion.choices[0].message.parsed
```

### Key Advantages
1. **Type Safety**: Pydantic models enforce schema at runtime
2. **Reliability**: OpenAI's structured output guarantees valid JSON
3. **Simplicity**: No manual JSON parsing, regex extraction, or quote fixing
4. **Consistency**: Both pipelines now use the same API pattern
5. **Debugging**: Optional fields like `reason` provide transparency

## Migration Impact

### No Breaking Changes
- All output file formats remain the same
- CSV outputs unchanged
- JSON structure preserved
- Backward compatible with existing workflows

### Improved Reliability
- Eliminates JSON parsing errors
- Removes fragile regex-based extraction
- Handles malformed responses gracefully
- Consistent error messages

## Testing Recommendations

1. **Run lec_summary_json.py**:
   ```bash
   cd lec_slide_base_lec_summary
   python lec_summary_json.py
   ```
   - Check for `[+]` success indicators
   - Verify output/cs_61a_lecture_topic_summaries.json

2. **Run predict_lec_summary.py**:
   ```bash
   python predict_lec_summary.py
   ```
   - Use `max_files=50` for testing (default parameter)
   - Verify output/cs_61a_video_lecture_eval.csv
   - Check prediction accuracy

3. **Compare with youtube_base_lec_summary**:
   - Both should show similar API patterns
   - Both should use Pydantic models
   - Both should produce structured outputs

## Files Modified
- [lec_summary_json.py](lec_summary_json.py)
- [predict_lec_summary.py](predict_lec_summary.py)

## References
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Pydantic Documentation: https://docs.pydantic.dev/
- Original Implementation: [../youtube_base_lec_summary/](../youtube_base_lec_summary/)

---
*Updated: January 28, 2026*
*Both pipelines now use consistent OpenAI structured output patterns*
