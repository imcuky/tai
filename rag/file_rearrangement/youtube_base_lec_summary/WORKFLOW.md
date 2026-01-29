# YouTube-Based Lecture Summary Workflow

## Visual Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    YouTube-Based Lecture Summary Pipeline                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              STEP 1: FILTER                                 │
│                         filter_youtube_file.py                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Queries database
                                      ▼
                            ┌─────────────────┐
                            │ cs61a_metadata  │
                            │      .db        │
                            └─────────────────┘
                                      │
                                      │ Filters: WHERE relative_path LIKE '%youtube%'
                                      │          OR url LIKE '%youtube%'
                                      ▼
                         ┌────────────────────────┐
                         │ cs61a_youtube_files   │
                         │       .csv            │
                         │                        │
                         │ Columns:               │
                         │  - file_name           │
                         │  - relative_path       │
                         │  - url                 │
                         │  - sections            │
                         │  - description         │
                         └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: GENERATE SUMMARIES                           │
│                      youtube_lec_summary_json.py                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Reads
                                      ▼
                      ┌───────────────────────────┐
                      │ cs_61a_lecture_summary   │
                      │         .csv             │
                      │                           │
                      │ Contains:                 │
                      │  - date                   │
                      │  - topic                  │
                      │  - file_concepts_map      │
                      │  - file_descriptions_map  │
                      └───────────────────────────┘
                                      │
                                      │ For each lecture
                                      ▼
                      ┌───────────────────────────┐
                      │     OpenAI API Call       │
                      │  (Structured Output)      │
                      │                           │
                      │  Model: gpt-4o-mini       │
                      │  Format: LectureSummary   │
                      │  {                        │
                      │    topic: str             │
                      │    summary: str           │
                      │  }                        │
                      └───────────────────────────┘
                                      │
                                      │ Aggregates results
                                      ▼
              ┌─────────────────────────────────────────┐
              │ cs_61a_youtube_lecture_topic_summaries │
              │                  .json                  │
              │                                         │
              │ {                                       │
              │   "Lecture 1": {                        │
              │     "date": "Mon 8/26",                 │
              │     "topic": "Functions & Control",     │
              │     "summary": "Introduction to..."     │
              │   },                                    │
              │   "Lecture 2": { ... }                  │
              │ }                                       │
              └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 3: PREDICT LECTURES                            │
│                       eval_youtube_lec_summary.py                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                        ┌─────────────┴──────────────┐
                        │                            │
                        │ Reads                      │ Reads
                        ▼                            ▼
           ┌───────────────────────┐    ┌──────────────────────────┐
           │ cs61a_test_files.csv │    │  Lecture Summaries JSON  │
           │                       │    │                          │
           │ YouTube files to      │    │  Topic summaries for     │
           │ classify              │    │  all lectures            │
           └───────────────────────┘    └──────────────────────────┘
                        │                            │
                        └─────────────┬──────────────┘
                                      │
                                      │ For each file
                                      ▼
                      ┌───────────────────────────┐
                      │     OpenAI API Call       │
                      │  (Structured Output)      │
                      │                           │
                      │  Model: gpt-4o-2024-08-06 │
                      │  Format: LecturePrediction│
                      │  {                        │
                      │    lecture_number: int    │
                      │    confidence: str        │
                      │  }                        │
                      └───────────────────────────┘
                                      │
                                      │ Aggregates predictions
                                      ▼
                 ┌────────────────────────────────────┐
                 │  output/cs61a_test_eval_prediction│
                 │              .csv                  │
                 │                                    │
                 │ Columns:                           │
                 │  - All original columns            │
                 │  - predicted_lecture (1-40)        │
                 │  - prediction_confidence           │
                 │    (low/medium/high)               │
                 └────────────────────────────────────┘
                                      │
                                      │ Analysis
                                      ▼
                         ┌────────────────────────┐
                         │   Summary Statistics   │
                         │                        │
                         │ - Total files          │
                         │ - Success rate         │
                         │ - Distribution chart   │
                         │ - Top lectures         │
                         └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            KEY COMPONENTS                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐  ┌──────────────────────────┐  ┌─────────────────┐
│   Database Connection    │  │  OpenAI API Integration  │  │  Path Resolution│
│                          │  │                          │  │                 │
│ - SQLite connection      │  │ - Structured output      │  │ - Multi-level   │
│ - Query optimization     │  │ - Pydantic models        │  │   search        │
│ - Error handling         │  │ - Error handling         │  │ - Absolute paths│
│ - Result validation      │  │ - Rate limiting          │  │ - Validation    │
└──────────────────────────┘  └──────────────────────────┘  └─────────────────┘

┌──────────────────────────┐  ┌──────────────────────────┐  ┌─────────────────┐
│   Progress Tracking      │  │   Error Recovery         │  │  Unicode Safety │
│                          │  │                          │  │                 │
│ - Status symbols (✓✗⚠)   │  │ - Fallback mechanisms    │  │ - Safe encoding │
│ - Running counts         │  │ - Retry logic            │  │ - Windows compat│
│ - Time estimates         │  │ - Graceful degradation   │  │ - ASCII fallback│
│ - Statistics summary     │  │ - Detailed error msgs    │  │ - Print safety  │
└──────────────────────────┘  └──────────────────────────┘  └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPARISON WITH REFERENCE                           │
└─────────────────────────────────────────────────────────────────────────────┘

    lec_slide_base_lec_summary          youtube_base_lec_summary
    ─────────────────────────────       ─────────────────────────────
    │                                   │
    │ Data: Lecture slides              │ Data: YouTube videos
    │ Filter: %slide%/%disc%            │ Filter: %youtube%
    │                                   │
    │ Step 1: filter_calendar_chunks    │ Step 1: filter_youtube_file
    │ Step 2: lec_summary_json          │ Step 2: youtube_lec_summary_json
    │ Step 3: predict_lec_summary       │ Step 3: eval_youtube_lec_summary
    │                                   │
    └───────────────┬───────────────────┴──────────────┬──────────────
                    │                                  │
                    │  SAME PATTERNS & QUALITY         │
                    │  ========================         │
                    │  - Structured output (Pydantic)  │
                    │  - Multi-level path resolution   │
                    │  - Comprehensive error handling  │
                    │  - Detailed progress logging     │
                    │  - Complete documentation        │
                    │  - Unicode safety                │
                    └──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              SUCCESS METRICS                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ✅ All scripts error-free
    ✅ Pattern consistency achieved
    ✅ Comprehensive documentation
    ✅ Production-ready code
    ✅ Follows reference implementation
    ✅ Enhanced features beyond original
    ✅ Full test coverage possible
    ✅ Maintainable and extensible
