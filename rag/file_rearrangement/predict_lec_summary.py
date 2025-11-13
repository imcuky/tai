import os
import json
import sqlite3
import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from openai import OpenAI


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
	parsed = _safe_json_or_literal_load(sections_raw)
	concepts = []
	seen = set()
	if isinstance(parsed, list):
		for sec in parsed:
			if isinstance(sec, dict) and 'key_concept' in sec:
				kc = sec.get('key_concept')
				if isinstance(kc, str):
					val = kc.strip()
					if val and val.lower() not in seen:
						seen.add(val.lower())
						concepts.append(val)
				elif isinstance(kc, list):
					for item in kc:
						if isinstance(item, str):
							val = item.strip()
							if val and val.lower() not in seen:
								seen.add(val.lower())
								concepts.append(val)
	return concepts


def load_youtube_videos(db_path: str = "cs61a_metadata.db") -> pd.DataFrame:
	if not os.path.exists(db_path):
		raise FileNotFoundError(f"DB not found: {db_path}")
	conn = sqlite3.connect(db_path)
	try:
		query = (
			"SELECT uuid, file_name, relative_path, url, sections FROM file "
			"WHERE lower(relative_path) LIKE '%youtub%' OR lower(url) LIKE '%youtube%'"
		)
		df = pd.read_sql_query(query, conn)
	finally:
		conn.close()
	# Add key_concepts column parsed from sections
	df['video_key_concepts'] = df['sections'].apply(parse_sections_key_concepts)
	return df


def load_lecture_topics_json(json_path: str = "cs_61a_lecture_topic_summaries.json") -> pd.DataFrame | None:
	"""Load generated lecture topics JSON. Returns DataFrame with lecture_number, topic_generated, summary_generated, date."""
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


def load_lecture_summaries(csv_path_main: str = "cs_61a_lecture_summary.csv",
						   json_topics_path: str = "cs_61a_lecture_topic_summaries.json") -> pd.DataFrame:
	# Prefer main; fall back to _alt if needed
	path = csv_path_main
	alt = os.path.splitext(csv_path_main)[0] + "_alt.csv"
	if not os.path.exists(path) and os.path.exists(alt):
		path = alt
	topics_df = load_lecture_topics_json(json_topics_path)

	if not os.path.exists(path):
		# If CSV missing, try to build from JSON topics only
		if topics_df is None:
			raise FileNotFoundError(f"Lecture summary CSV not found: {csv_path_main} (or {alt}) and topics JSON missing: {json_topics_path}")
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

	df['key_concepts_list'] = df.get('key_concepts', '').apply(parse_list)
	df['aspects_list'] = df.get('aspects', '').apply(parse_list)
	df['slide_files_list'] = df.get('slide_files', '').apply(parse_list)
	df['lecture_number'] = df.index + 1

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
	# Also build file basename -> category label (e.g., "Mutability (Su25)")
	file_to_category: dict[str, str] = {}
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
				# Derive category label from the path: segment after 'youtubeXX'
				try:
					parts = re.split(r"[\\/]+", path)
					# find index of a segment like 'youtube10'
					yt_idx = next((i for i, p in enumerate(parts) if re.match(r"youtube\d+", p, re.IGNORECASE)), None)
					if yt_idx is not None and yt_idx + 1 < len(parts):
						cat_label = parts[yt_idx + 1]
						if base and base not in file_to_category:
							file_to_category[base] = cat_label
						if simple and simple not in file_to_category:
							file_to_category[simple] = cat_label
				except Exception:
					pass
	# Return both maps via a combined dict for backward compatibility: attach under special key '__categories__'
	file_to_lecture['__categories__'] = file_to_category
	return file_to_lecture


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


def _topic_for_lecture(lectures_df: pd.DataFrame, number: int) -> str:
    try:
        row = lectures_df.loc[lectures_df['lecture_number'] == number].iloc[0]
    except Exception:
        return ""
    return (row.get('topic_generated') or row.get('topic') or "")


def jaccard_similarity(a: set, b: set) -> float:
	if not a and not b:
		return 0.0
	inter = len(a & b)
	union = len(a | b)
	return inter / union if union else 0.0


def choose_by_baseline(video_concepts: list, lectures_df: pd.DataFrame):
	vset = set([c.lower() for c in video_concepts if isinstance(c, str)])
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
		score = jaccard_similarity(vset, lset)
		if score > best_score:
			best_score = score
			best_num = int(lec['lecture_number'])
	return best_num, best_score


def call_openai_classify(client: OpenAI, video_meta: dict, lectures: list, model: str = "gpt-4o-mini"):
	# Build a compact lecture list for the prompt
	lecture_briefs = []
	for lec in lectures:
		lecture_briefs.append({
			"number": lec['lecture_number'],
			"date": lec.get('date', ''),
			# Prefer generated topic/summary when available
			"topic": lec.get('topic_generated') or lec.get('topic', ''),
			"summary": lec.get('summary_generated', ''),
			"key_concepts": lec.get('key_concepts_list', [])[:20],
		})

	user_prompt = (
		"You are given a set of lecture summaries and a CS 61A video file's metadata.\n"
		"Pick the single most likely lecture number that this video belongs to.\n"
		"Return ONLY JSON with keys \"lecture_number\" (integer), \"confidence\" (0-1), and \"reason\". No markdown.\n\n"
		f"Lectures: {json.dumps(lecture_briefs, ensure_ascii=False)}\n\n"
		f"Video: {json.dumps(video_meta, ensure_ascii=False)}\n\n"
	"Rules: Use key_concepts alignment primarily; compare with lecture topics/summaries; filenames/paths/urls may hint."
	)

	try:
		resp = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": "Output strict JSON only. Keys: lecture_number (int), confidence (0-1), reason (short)."},
				{"role": "user", "content": user_prompt},
			],
			temperature=0.1,
			max_tokens=220,
		)
		content = (resp.choices[0].message.content or '').strip()
		# Parse robustly
		try:
			parsed = json.loads(content)
		except Exception:
			if content.startswith("```"):
				content = re.sub(r"^```[a-zA-Z]*\n|```$", "", content).strip()
			m = re.search(r"\{[\s\S]*\}", content)
			raw_obj = m.group(0) if m else content
			try:
				parsed = json.loads(raw_obj)
			except Exception:
				import ast
				try:
					parsed = ast.literal_eval(raw_obj)
				except Exception:
					parsed = {}
		ln = parsed.get('lecture_number')
		conf = parsed.get('confidence', 0)
		reason = parsed.get('reason', '')
		try:
			ln = int(ln) if ln is not None else None
		except Exception:
			ln = None
		try:
			conf = float(conf)
		except Exception:
			conf = 0.0
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


def main():
	print("Evaluating CS 61A lecture classification for YouTube videos (groundtruth JSON based)")

	# Data loads
	videos_df = load_youtube_videos("cs61a_metadata.db")
	lectures_df = load_lecture_summaries("cs_61a_lecture_summary.csv")
	gt_calendar_map = extract_calendar_video_map("cs_61a_calendar_with_paths.csv")  # legacy
	gt_file_map = load_groundtruth_json("groundtruth_youtube_only.json")
	# Extract category mapping embedded under special key
	gt_category_map = {}
	if '__categories__' in gt_file_map:
		gt_category_map = gt_file_map.pop('__categories__') or {}

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

	results = []
	for _, v in videos_df.iterrows():
		file_name = v.get('file_name', '') or ''
		file_name_lower = file_name.lower()
		rel_path = (v.get('relative_path', '') or '').lower()

		vid_meta = {
			"uuid": v.get('uuid', ''),
			"file_name": file_name,
			"relative_path": v.get('relative_path', ''),
			"url": v.get('url', ''),
			"key_concepts": v.get('video_key_concepts', []),
		}

		base_num, base_score = choose_by_baseline(vid_meta['key_concepts'], lectures_df)

		pred_num, pred_conf, pred_reason = (None, 0.0, 'no-openai')
		if client is not None:
			pred_num, pred_conf, pred_reason = call_openai_classify(client, vid_meta, lectures_list)

		# Ground truth from JSON file mapping (primary)
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

		# Ground truth category label from JSON or path
		gt_category = None
		if file_name_lower in gt_category_map:
			gt_category = gt_category_map[file_name_lower]
		else:
			if simplified in gt_category_map:
				gt_category = gt_category_map[simplified]
			else:
				# Try to derive from relative_path using youtube segment
				parts = re.split(r"[\\/]+", rel_path)
				try:
					yt_idx = next((i for i, p in enumerate(parts) if re.match(r"youtube\d+", p, re.IGNORECASE)), None)
					if yt_idx is not None and yt_idx + 1 < len(parts):
						gt_category = parts[yt_idx + 1]
				except Exception:
					pass

		# Legacy calendar ground truth for reference (video id)
		video_id = extract_video_id(vid_meta['url'])
		gt_calendar_num = gt_calendar_map.get(video_id)

		# Use gt_json_num as authoritative for numeric accuracy; fall back to calendar if absent
		authoritative_gt = gt_json_num if gt_json_num is not None else gt_calendar_num

		# Compute category-aware correctness using predicted lecture topic
		pred_topic = _topic_for_lecture(lectures_df, pred_num) if pred_num else ""
		base_topic = _topic_for_lecture(lectures_df, base_num) if base_num else ""
		pred_cat_match = _category_match(gt_category or "", pred_topic)
		base_cat_match = _category_match(gt_category or "", base_topic)

		results.append({
			# "video_uuid": vid_meta['uuid'],
			"file_name": file_name,
			"url": vid_meta['url'],
			"video_id": video_id or '',
			"predicted_lecture": pred_num if pred_num is not None else '',
			"predicted_confidence": pred_conf,
			"predicted_reason": pred_reason[:400],
			# "baseline_lecture": base_num if base_num is not None else '',
			# "baseline_score": round(base_score, 4),
			"ground_truth_lecture_json": gt_json_num if gt_json_num is not None else '',
			"ground_truth_lecture_calendar": gt_calendar_num if gt_calendar_num is not None else '',
			"ground_truth_lecture": authoritative_gt if authoritative_gt is not None else '',
			"ground_truth_category": gt_category or '',
			"pred_topic": pred_topic,
			# "base_topic": base_topic,
			"pred_correct_numeric": (authoritative_gt is not None and pred_num == authoritative_gt),
			# "base_correct_numeric": (authoritative_gt is not None and base_num == authoritative_gt),
			"pred_correct_by_category": bool(pred_cat_match),
			# "base_correct_by_category": bool(base_cat_match),
			"pred_correct": (authoritative_gt is not None and pred_num == authoritative_gt) or pred_cat_match,
			"base_correct": (authoritative_gt is not None and base_num == authoritative_gt) or base_cat_match,
		})

	out_df = pd.DataFrame(results)
	out_path = "cs_61a_video_lecture_eval.csv"
	out_df.to_csv(out_path, index=False)

	total = len(out_df)
	with_gt_numeric = out_df['ground_truth_lecture'].astype(str).ne('').sum()
	with_gt_any = ((out_df['ground_truth_lecture'].astype(str).ne('')) | (out_df['ground_truth_category'].astype(str).ne(''))).sum()
	pred_correct_numeric = out_df['pred_correct_numeric'].sum()
	base_correct_numeric = out_df['base_correct_numeric'].sum()
	pred_correct_any = out_df['pred_correct'].sum()
	base_correct_any = out_df['base_correct'].sum()
	print(f"Total videos: {total}")
	print(f"With numeric GT: {with_gt_numeric}")
	if with_gt_numeric:
		print(f"OpenAI acc (strict numeric): {pred_correct_numeric}/{with_gt_numeric} ({(pred_correct_numeric/with_gt_numeric)*100:.1f}%)")
		print(f"Baseline acc (strict numeric): {base_correct_numeric}/{with_gt_numeric} ({(base_correct_numeric/with_gt_numeric)*100:.1f}%)")
	print(f"With numeric or category GT: {with_gt_any}")
	if with_gt_any:
		print(f"OpenAI acc (category-aware): {pred_correct_any}/{with_gt_any} ({(pred_correct_any/with_gt_any)*100:.1f}%)")
		print(f"Baseline acc (category-aware): {base_correct_any}/{with_gt_any} ({(base_correct_any/with_gt_any)*100:.1f}%)")
	print(f"Results saved to: {out_path}")


if __name__ == "__main__":
	main()

