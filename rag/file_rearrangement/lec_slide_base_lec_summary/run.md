# Step 1: Generate lecture summary from slide files
python generate_slide_summary_csv.py

# Step 2: Generate OpenAI topic summaries
python lec_summary_json.py

# Step 3: Test predictions (50 files)
python predict_lec_summary.py