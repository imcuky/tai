import os
import json
import sqlite3
import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class LecturePrediction(BaseModel):
    """Structured output model for lecture predictions."""
    lecture_number: int
    reason: str = ""  # Optional: explanation for the prediction


def _safe_json_or_literal_load(s):
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


def parse_sections_key_concepts(sections_raw):
	"""Parse key_concepts from sections, merging with Definition aspect if available."""
	parsed = _safe_json_or_literal_load(sections_raw)
	concepts = []
	seen = set()
	if isinstance(parsed, list):
		for sec in parsed:
			if isinstance(sec, dict):
				# Get key_concept name(s)
				kc_names = []
				kc = sec.get('key_concept')
				if isinstance(kc, str):
					val = kc.strip()
					if val:
						kc_names.append(val)
				elif isinstance(kc, list):
					for item in kc:
						if isinstance(item, str):
							val = item.strip()
							if val:
								kc_names.append(val)
				
				# Look for Definition in aspects to merge
				definition_content = None
				asp_list = sec.get('aspects')
				if isinstance(asp_list, list):
					for a in asp_list:
						if isinstance(a, dict):
							atype = str(a.get('type', '')).strip()
							content = str(a.get('content', '')).strip()
							if atype.lower() == 'definition' and content:
								definition_content = content
								break
				
				# Add merged key concepts
				for name in kc_names:
					if definition_content:
						final_val = f"{name}: {definition_content}"
					else:
						final_val = name
					if final_val.lower() not in seen:
						seen.add(final_val.lower())
						concepts.append(final_val)
	return concepts


def load_eval_files(db_path: str = "cs61a_metadata.db") -> pd.DataFrame:
	if not os.path.exists(db_path):
		raise FileNotFoundError(f"DB not found: {db_path}")
	conn = sqlite3.connect(db_path)
	try:
		# Construct WHERE clause for youtube, disc (discussion), textbook
		filters = [
			"lower(relative_path) LIKE '%youtub%'",
			"lower(relative_path) LIKE '%disc%'",
			"lower(relative_path) LIKE '%textbook%'",
			"lower(url) LIKE '%youtube%'",
		]
		clause = " OR ".join(filters)
		
		# Prefer including file_path if available
		try:
			query = (
				f"SELECT file_name, relative_path, url, sections, description FROM file "
				f"WHERE {clause} OR lower(file_path) LIKE '%youtube%'"
			)
			df = pd.read_sql_query(query, conn)
		except Exception:
			query = (
				f"SELECT file_name, relative_path, url, sections, description FROM file "
				f"WHERE {clause}"
			)
			df = pd.read_sql_query(query, conn)
			df['file_path'] = ''
	finally:
		conn.close()
	# Add key_concepts/description/category parsed from sections/paths

	df['video_key_concepts'] = df['sections'].apply(parse_sections_key_concepts)
	
	return df


def load_lecture_topics_json(json_path: str = "output/cs_61a_lecture_topic_summaries.json") -> pd.DataFrame | None:
	"""Load generated lecture topics JSON. Returns DataFrame with lecture_number, topic_generated, summary_generated, date."""
	# Resolve path relative to script dir if needed
	if not os.path.exists(json_path):
		script_dir = os.path.dirname(os.path.abspath(__file__))
		json_path = os.path.join(script_dir, "output", "cs_61a_lecture_topic_summaries.json")
	
	if not os.path.exists(json_path):
		return None
	try:
		with open(json_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		rows = []
		for key, val in data.items():
			# key like "Lecture 1"
			try:
				num = int(re.findall(r"\d+", key)[0])
			except Exception:
				continue
			rows.append({
				'lecture_number': num,
				'date': val.get('date', ''),
				'topic_generated': val.get('topic', ''),
				'summary_generated': val.get('summary', ''),
			})
		if not rows:
			return None
		df = pd.DataFrame(rows).sort_values('lecture_number').reset_index(drop=True)
		return df
	except Exception:
		return None


def load_lecture_summaries(csv_path_main: str = "output/cs_61a_lecture_summary.csv",
						   json_topics_path: str = "output/cs_61a_lecture_topic_summaries.json") -> pd.DataFrame:
	# Resolve paths relative to script dir if needed
	if not os.path.exists(csv_path_main):
		script_dir = os.path.dirname(os.path.abspath(__file__))
		csv_path_main = os.path.join(script_dir, "output", os.path.basename(csv_path_main))

	# Prefer main; fall back to _alt if needed
	path = csv_path_main
	alt = os.path.splitext(csv_path_main)[0] + "_alt.csv"
	if not os.path.exists(path) and os.path.exists(alt):
		path = alt
	
	topics_df = load_lecture_topics_json(json_topics_path)

	if not os.path.exists(path):
		# If CSV missing, try to build from JSON topics only
		if topics_df is None:
			print(f"Warning: Summary CSV not found at {path} and JSON topics missing.")
			# Return empty structure as last resort mechanism
			return pd.DataFrame(columns=['lecture_number', 'key_concepts_list', 'aspects_list', 'slide_files_list'])
		
		df = topics_df.copy()
		# Create empty placeholders for lists used downstream
		df['key_concepts_list'] = [[] for _ in range(len(df))]
		df['aspects_list'] = [[] for _ in range(len(df))]
		df['slide_files_list'] = [[] for _ in range(len(df))]
		return df

	df = pd.read_csv(path)

	def parse_list(s):
		v = _safe_json_or_literal_load(s)
		return v if isinstance(v, list) else []

	# Safely handle missing columns by returning empty lists series
	def safe_apply_list(col_name):
		if col_name in df.columns:
			return df[col_name].apply(parse_list)
		return pd.Series([[] for _ in range(len(df))])

	df['key_concepts_list'] = safe_apply_list('key_concepts')
	df['aspects_list'] = safe_apply_list('aspects')
	df['slide_files_list'] = safe_apply_list('slide_files')
	
	# Use existing lecture_number if available, else assume index+1
	if 'lecture_number' not in df.columns:
		df['lecture_number'] = df.index + 1

	# Optional new maps for categories and descriptions per file
	def parse_map(s):
		v = _safe_json_or_literal_load(s)
		return v if isinstance(v, dict) else {}
	
	def safe_apply_map(col_name):
		if col_name in df.columns:
			return df[col_name].apply(parse_map)
		return pd.Series([{} for _ in range(len(df))])

	df['file_categories_map'] = safe_apply_map('file_categories_map')
	df['file_descriptions_map'] = safe_apply_map('file_descriptions_map')

	# Merge in generated topics/summary if available
	if topics_df is not None:
		df = df.merge(topics_df, on='lecture_number', how='left', suffixes=('', '_gen'))
	return df


def extract_calendar_video_map(calendar_csv: str = "cs_61a_calendar_with_paths.csv") -> dict:
	"""(Legacy) Map youtube video_id -> lecture_number from calendar; retained for comparison."""
	if not os.path.exists(calendar_csv):
		return {}
	df = pd.read_csv(calendar_csv)
	video_to_lecture = {}
	for idx, row in df.iterrows():
		original = str(row.get('original_text', '') or '')
		urls = re.findall(r'https?://[^\s\)\"]+', original)
		md_urls = re.findall(r'\[[^\]]*\]\(([^\)]*)\)', original)
		urls.extend(md_urls)
		for u in urls:
			try:
				parsed = urlparse(u)
				if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
					vid = None
					if 'youtube.com' in parsed.netloc:
						q = parse_qs(parsed.query)
						vid = (q.get('v') or [None])[0]
					else:
						parts = [p for p in parsed.path.split('/') if p]
						if parts:
							vid = parts[-1]
					if vid and vid not in video_to_lecture:
						video_to_lecture[vid] = idx + 1
			except Exception:
				continue
	return video_to_lecture


def load_groundtruth_json(gt_path: str = "groundtruth_youtube_only.json") -> dict:
	"""Load ground truth mapping JSON (lecture directories -> file lists) and
	build a map basename_lower -> lecture_number.

	JSON structure example:
	  {
		"lec02": { "lec02\\youtube02": ["lec02\\youtube02\\Functions\\1-Welcome.webm.json.txt", ...] },
		...
	  }
	We interpret top-level key 'lecXX' as lecture number XX.
	For each listed path we take the final component (basename) lowercased as a key.
	Duplicate basenames mapping to conflicting lectures will keep the first assignment.
	Returns dict: { basename_lower: lecture_number }.
	"""
	if not os.path.exists(gt_path):
		print(f"Ground truth JSON not found: {gt_path}")
		return {}
	try:
		with open(gt_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except Exception as e:
		print(f"Failed to load ground truth JSON: {e}")
		return {}

	file_to_lecture: dict[str, int] = {}
	lec_key_pattern = re.compile(r'lec(\d+)', re.IGNORECASE)
	for lec_key, nested in data.items():
		m = lec_key_pattern.search(lec_key)
		if not m:
			continue
		lec_num = int(m.group(1))
		if not isinstance(nested, dict):
			continue
		for _subdir, file_list in nested.items():
			if not isinstance(file_list, list):
				continue
			for path in file_list:
				if not isinstance(path, str):
					continue
				base = os.path.basename(path).lower()
				if base and base not in file_to_lecture:
					file_to_lecture[base] = lec_num
				# Add simplified variant stripping composite extensions for robustness
				simple = re.sub(r'(\.json\.txt|_metadata\.yaml\.txt|\.txt|\.json)$', '', base)
				if simple and simple not in file_to_lecture:
					file_to_lecture[simple] = lec_num
	return file_to_lecture


def derive_gt_from_file_path(file_path: str) -> int | None:
	"""Derive numeric lecture from a canonical file_path like
	'CS 61A/study/lecture/lec03/youtube03/Control/1-Multiple Environments.webm'.
	Returns lecture_number or None.
	"""
	if not file_path or not isinstance(file_path, str):
		return None
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
	return lec_num


def jaccard_similarity(a: set, b: set) -> float:
	if not a and not b:
		return 0.0
	inter = len(a & b)
	union = len(a | b)
	return inter / union if union else 0.0


def choose_by_baseline(video_concepts: list, lectures_df: pd.DataFrame, video_descs: list | None = None):
	vset = set([c.lower() for c in video_concepts if isinstance(c, str)])
	# include description tokens from video
	if video_descs:
		for d in video_descs:
			for w in re.findall(r"[A-Za-z0-9_]+", str(d)):
				vset.add(w.lower())
	best_num, best_score = None, -1.0
	for _, lec in lectures_df.iterrows():
		# Prefer key concepts if present; else fall back to words in generated summary/topic
		lec_kc = lec.get('key_concepts_list') or []
		if lec_kc:
			lset = set([c.lower() for c in lec_kc if isinstance(c, str)])
		else:
			text = ' '.join([
				str(lec.get('topic_generated', '') or ''),
				str(lec.get('summary_generated', '') or ''),
				str(lec.get('topic', '') or ''),
			])
			# simple tokenization: letters/digits words
			lset = set([w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text)])
		# include lecture file descriptions
		fdesc = lec.get('file_descriptions_map') or {}
		if isinstance(fdesc, dict):
			for arr in fdesc.values():
				if isinstance(arr, list):
					for d in arr:
						for w in re.findall(r"[A-Za-z0-9_]+", str(d)):
							lset.add(w.lower())
		score = jaccard_similarity(vset, lset)
		if score > best_score:
			best_score = score
			best_num = int(lec['lecture_number'])
	return best_num, best_score


def call_openai_classify(client: OpenAI, video_meta: dict, lectures: list, model: str = "gpt-4o-mini"):
	# Build a compact lecture list for the prompt
	lecture_briefs = []
	for lec in lectures:
		# Aggregate category stats and a few description snippets
		cat_map = lec.get('file_categories_map') or {}
		cats = []
		if isinstance(cat_map, dict):
			cats = [c for c in cat_map.values() if isinstance(c, str)]
		from collections import Counter
		cat_counts = Counter([c.lower() for c in cats])
		top_cats = cat_counts.most_common(3)

		fdesc = lec.get('file_descriptions_map') or {}
		desc_snips = []
		if isinstance(fdesc, dict):
			for arr in fdesc.values():
				if isinstance(arr, list):
					for d in arr:
						if isinstance(d, str) and d.strip():
							desc_snips.append(d.strip())
							if len(desc_snips) >= 3:
								break
				if len(desc_snips) >= 3:
					break

		lecture_briefs.append({
			"number": lec['lecture_number'],
			"date": lec.get('date', ''),
			# Prefer generated topic/summary when available
			"topic": lec.get('topic_generated') or lec.get('topic', ''),
			"summary": lec.get('summary_generated', ''),
			"key_concepts": lec.get('key_concepts_list', [])[:20],
			"categories_top": top_cats,
			"descriptions": desc_snips,
		})

	user_prompt = (
		"You are given a set of lecture summaries and video file's metadata.\n"
		"Pick the single most likely lecture number that this video belongs to.\n"
		"Use: key_concepts, category hints (e.g., slides/lecture/video), and description snippets.\n"
		f"Lectures: {json.dumps(lecture_briefs, ensure_ascii=False)}\n\n"
		f"Video: {json.dumps(video_meta, ensure_ascii=False)}\n\n"
		"Rules: Prioritize semantic alignment of concepts; corroborate with categories and brief descriptions."
	)

	try:
		# Use structured output API for reliable responses
		completion = client.beta.chat.completions.parse(
			model=model,
			messages=[
				{"role": "system", "content": "You are an expert at classifying CS study materials into course lecture categories. Analyze the lecture summaries and video metadata to determine the best match."},
				{"role": "user", "content": user_prompt},
			],
			response_format=LecturePrediction,
			temperature=0.1,
			max_tokens=300,
		)
		
		parsed_response = completion.choices[0].message.parsed
		if parsed_response:
			ln = parsed_response.lecture_number
			conf = getattr(parsed_response, 'confidence', 'medium')
			reason = getattr(parsed_response, 'reason', '')
		else:
			ln = None
			conf = 'low'
			reason = 'Parsing failed'
		
		# Ensure ln is an integer
		try:
			ln = int(ln) if ln is not None else None
		except Exception:
			ln = None
		
		# Convert confidence string to float
		# if isinstance(conf, str):
		# 	conf_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
		# 	conf = conf_map.get(conf.lower(), 0.5)
		# else:
		# 	try:
		# 		conf = float(conf)
		# 	except Exception:
		# 		conf = 0.5
		
		return ln, conf, str(reason)
	except Exception as e:
		return None, 0.0, f"OpenAI error: {e}"


def extract_video_id(url: str) -> str | None:
	if not url:
		return None
	try:
		p = urlparse(url)
		if 'youtube.com' in p.netloc:
			q = parse_qs(p.query)
			return (q.get('v') or [None])[0]
		if 'youtu.be' in p.netloc:
			parts = [x for x in p.path.split('/') if x]
			return parts[-1] if parts else None
	except Exception:
		return None
	return None


def main(max_files=None):
	"""
	Main function to evaluate CS 61A lecture classification for YouTube/Disc/Textbook files.
	
	Args:
		max_files: Maximum number of files to process (None for all). Use for testing.
	"""
	print("Evaluating CS 61A lecture classification for YouTube/Disc/Textbook files (groundtruth JSON based)")

	# Data loads
	videos_df = load_eval_files("cs61a_metadata.db")
	
	# Limit to subset for testing if specified
	if max_files is not None and max_files > 0:
		original_count = len(videos_df)
		videos_df = videos_df.head(max_files)
		print(f"\nLimited to first {len(videos_df)} files out of {original_count} for testing.")
	

	# Load from output directory (defaults handled in function)
	lectures_df = load_lecture_summaries()
	
	gt_calendar_map = extract_calendar_video_map("cs_61a_calendar_with_paths.csv")  # legacy
	gt_file_map = load_groundtruth_json("groundtruth_youtube_only.json")

	# Prepare LLM client
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	client = None
	if api_key:
		try:
			client = OpenAI(api_key=api_key)
			print("OpenAI client initialized.")
		except Exception as e:
			print(f"Warning: failed to init OpenAI: {e}")

	lectures_list = lectures_df.to_dict(orient='records')
	
	print(f"\nProcessing {len(videos_df)} video files...")
	print("=" * 60)

	results = []
	for idx, (_, v) in enumerate(videos_df.iterrows(), 1):
		file_name = v.get('file_name', '') or ''
		file_name_lower = file_name.lower()
		rel_path = (v.get('relative_path', '') or '').lower()

		# need to tell LLM how to understand the file_path
		# "CS 61A/study/lecture/lec03/youtube03/Control/1-Multiple Environments.webm"
		vid_meta = {
			"uuid": str(v.get('uuid', '')),
			"file_name": file_name,
			"relative_path": str(v.get('relative_path', '')),
			"file_path": str(v.get('file_path', '')), 
			"url": str(v.get('url', '')),
			"key_concepts": v.get('video_key_concepts', []),
			"descriptions": str(v.get('description', '')).strip(),
			# "category": v.get('video_category', ''), #
		}

		base_num, base_score = choose_by_baseline(vid_meta['key_concepts'], lectures_df, [vid_meta['descriptions']])

		pred_num, pred_conf, pred_reason = (None, 0.0, 'no-openai')
		if client is not None:
			pred_num, pred_conf, pred_reason = call_openai_classify(client, vid_meta, lectures_list)

		# Ground truth from file_path (primary)
		file_path = v.get('relative_path', '')
		gt_path_num = derive_gt_from_file_path(file_path)

		# Legacy JSON/name-based ground truth (fallback for numeric)
		gt_json_num = None
		if file_name_lower in gt_file_map:
			gt_json_num = gt_file_map[file_name_lower]
		else:
			# Try simplified variant removal of multi-extensions
			simplified = re.sub(r'(\.json\.txt|_metadata\.yaml\.txt|\.txt|\.json)$', '', file_name_lower)
			if simplified in gt_file_map:
				gt_json_num = gt_file_map[simplified]
			else:
				# Extract lecture hint from relative_path like 'lec05'
				m = re.search(r'lec(\d+)', rel_path)
				if m:
					gt_json_num = int(m.group(1))

		# Legacy calendar ground truth for reference (video id)
		video_id = extract_video_id(vid_meta['url'])
		gt_calendar_num = gt_calendar_map.get(video_id)

		# Use file_path-derived numeric first; fallback to JSON/rel_path; else calendar
		authoritative_gt = (
			gt_path_num if gt_path_num is not None else (
				gt_json_num if gt_json_num is not None else gt_calendar_num
			)
		)

		# Progress indicator
		safe_name = file_name.encode('ascii', errors='replace').decode('ascii')
		status = "[+]" if pred_num else "[X]"
		print(f"  {status} [{idx}/{len(videos_df)}] {safe_name[:50]:50s} -> Lecture {pred_num or -1:2d}")

		results.append({
			"file_name": file_name,
			"url": vid_meta['url'],
			"file_path": file_path,
			"predicted_lecture": pred_num if pred_num is not None else '',
			"predicted_reason": pred_reason[:400],
			"ground_truth_lecture_json": gt_json_num if gt_json_num is not None else '',
			"ground_truth_lecture_calendar": gt_calendar_num if gt_calendar_num is not None else '',
			"ground_truth_lecture": authoritative_gt if authoritative_gt is not None else '',
			"pred_correct": (authoritative_gt is not None and pred_num == authoritative_gt)
		})

	out_df = pd.DataFrame(results)
	
	# Create output directory
	current_script_dir = os.path.dirname(os.path.abspath(__file__))
	output_dir = os.path.join(current_script_dir, "output")
	os.makedirs(output_dir, exist_ok=True)
	
	# Save simplified output with only 3 columns: file_path (as relative_path), file_name, predicted_lecture
	out_path = os.path.join(output_dir, "cs_61a_video_lecture_eval.csv")
	
	# Calculate and print stats before simplification
	print_summary_stats_df(out_df)
	
	df_simplified = out_df[['file_path', 'file_name', 'predicted_lecture']].copy()
	df_simplified.rename(columns={'file_path': 'relative_path'}, inplace=True)
	df_simplified.to_csv(out_path, index=False)
	print(f"Results saved to: {out_path}")

def print_summary_stats_df(out_df: pd.DataFrame):
	# Filter to rows with non-empty numeric ground truth lecture only
	if 'ground_truth_lecture' not in out_df.columns:
		print("Ground truth columns missing for stats.")
		return

	mask = out_df['ground_truth_lecture'].notna() & out_df['ground_truth_lecture'].astype(str).str.strip().ne('')
	eval_df = out_df[mask].copy()

	total = len(out_df)
	with_gt = len(eval_df)
	pred_correct_any = int(eval_df['pred_correct'].sum()) if with_gt else 0

	print("\n" + "="*30)
	print("SUMMARY STATISTICS")
	print("="*30)
	print(f"Total videos: {total}")
	print(f"With numeric GT (filtered): {with_gt}")
	if with_gt:
		print(f"Accuracy (exact match to GT): {pred_correct_any}/{with_gt} = {pred_correct_any / with_gt:.2%}")