import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List

from openai import OpenAI


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REARRANGE_DIR = os.path.dirname(CURRENT_DIR)
if REARRANGE_DIR not in sys.path:
    sys.path.insert(0, REARRANGE_DIR)

from file_rearrang import BackboneGroup, OrphanMatchResponse  # noqa: E402


def load_json_file(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_child_folder(node: Dict, folder_name: str) -> Dict | None:
    children = node.get("children", {})
    if isinstance(children, dict):
        return children.get(folder_name)
    if isinstance(children, list):
        for child in children:
            if child.get("name") == folder_name and child.get("type") == "folder":
                return child
    return None


def fetch_file_description(cursor: sqlite3.Cursor, file_path: str, file_name: str) -> str:
    cursor.execute(
        "SELECT description FROM file WHERE relative_path = ? LIMIT 1",
        (file_path,),
    )
    row = cursor.fetchone()
    if row and row[0]:
        return row[0].strip()

    cursor.execute(
        "SELECT description FROM file WHERE file_name = ? LIMIT 1",
        (file_name,),
    )
    row = cursor.fetchone()
    return row[0].strip() if row and row[0] else ""


def build_lab_orphans(lab_node: Dict, db_path: str) -> List[Dict]:
    orphans: List[Dict] = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def recurse(folder_node: Dict, folder_path: str) -> None:
        files = folder_node.get("files", {})
        children = folder_node.get("children", {})

        file_nodes = list(files.values()) if isinstance(files, dict) else []
        child_folders = list(children.values()) if isinstance(children, dict) else []

        # Match main pipeline behavior: treat leaf folders with files as a single unit,
        # but do not collapse sequential structural folders (e.g., top-level lab).
        is_sequential = bool(folder_node.get("by_sequence", False))
        should_aggregate = bool(file_nodes) and not child_folders and not is_sequential

        if should_aggregate:
            part_names: List[str] = []
            part_details: List[str] = []
            for file_node in file_nodes:
                file_name = file_node.get("name", "")
                file_rel_path = file_node.get("path", "")
                if file_name:
                    part_names.append(file_name)
                desc = fetch_file_description(cursor, file_rel_path, file_name)
                if desc:
                    part_details.append(f"{file_name}: {desc}")

            combined_desc = f"Folder containing: {', '.join(part_names)}"
            if part_details:
                combined_desc += ". Details: " + " | ".join(part_details)

            orphans.append(
                {
                    "structure_path": folder_path,
                    "relative_path": folder_path,
                    "name": folder_node.get("name", folder_path),
                    "type": "folder_unit",
                    "description": combined_desc,
                }
            )
            return

        # Keep top-level files as direct orphan files.
        for file_node in file_nodes:
            file_name = file_node.get("name", "")
            file_rel_path = file_node.get("path", "")
            orphans.append(
                {
                    "structure_path": file_rel_path,
                    "relative_path": file_rel_path,
                    "name": file_name,
                    "type": "file",
                    "description": fetch_file_description(cursor, file_rel_path, file_name),
                }
            )

        for child in child_folders:
            child_name = child.get("name", "")
            if not child_name:
                continue
            recurse(child, f"{folder_path}/{child_name}")

    try:
        recurse(lab_node, "lab")
    finally:
        conn.close()

    return orphans


def run_direct_matching(
    backbone_path: str,
    groups: List[BackboneGroup],
    orphans: List[Dict],
    chunk_size: int = 50,
) -> OrphanMatchResponse:
    client = OpenAI()

    groups_summary = [g.model_dump() for g in groups]
    all_matches = []

    for i in range(0, len(orphans), chunk_size):
        batch = orphans[i : i + chunk_size]
        completion = client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent course material organizer for any subject "
                        "(Computer Science, Math, Literature, etc.).\n\n"
                        f"The folder '{backbone_path}' defines the chronological 'backbone' of this course.\n"
                        "You will receive:\n"
                        "- A list of 'Existing Groups' (the backbone units) with their descriptions.\n"
                        "- A batch of 'Orphan Files' that need to be categorized.\n\n"
                        "Your Task:\n"
                        "For EACH orphan, assign it to its relevant group. "
                        "If you think this material can match multiple groups, match ALL of them.\n\n"
                        "Topic-Only Mapping Rule (Critical):\n"
                        "- Assign based on actual lecture topic coverage only (concepts/skills in the orphan description and group description).\n"
                        "- Do NOT assign by assessment stage words such as 'review', 'midterm', 'final', 'exam', or 'discussion' alone.\n"
                        "- 'Final Review' content is NOT automatically 'Midterm Review'; map it only to lectures whose topics are explicitly covered.\n\n"
                        "Matching Considerations:\n"
                        "1. Strong Match (Preferred): if the file's name or description strongly "
                        "relates to one or more backbone groups.\n"
                        "2. Ambiguous/No Match (Fallback): place it in 'Lecture Miscellaneous'.\n\n"
                        "Constraints:\n"
                        "- Use existing group_name exactly as provided.\n"
                        "- Every orphan MUST be assigned to at least one group.\n"
                        "- Do NOT create files that are not in the orphans list."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "backbone_folder": backbone_path,
                            "existing_groups": groups_summary,
                            "orphans": batch,
                        },
                        indent=2,
                    ),
                },
            ],
            response_format=OrphanMatchResponse,
            seed=42,
        )

        parsed = completion.choices[0].message.parsed
        valid_paths = {o["relative_path"] for o in batch}
        for match in parsed.matches:
            if match.item_path.strip() in valid_paths:
                all_matches.append(match)

    return OrphanMatchResponse(matches=all_matches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test direct multi-match using only LAB orphans against existing backbone groups"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(REARRANGE_DIR, "input", "bfs_v3_tree_61.json"),
        help="Path to full tree JSON.",
    )
    parser.add_argument(
        "--db",
        default=os.path.join(REARRANGE_DIR, "input", "CS_61A_metadata_NewPT.db"),
        help="Path to SQLite metadata DB.",
    )
    parser.add_argument(
        "--groups",
        default=os.path.join(REARRANGE_DIR, "logs", "02_2_backbone_groups.json"),
        help="Path to existing backbone groups JSON.",
    )
    parser.add_argument(
        "--backbone",
        default="study/assets/slides",
        help="Backbone folder path label used in prompt.",
    )
    parser.add_argument(
        "--orphans-out",
        default=os.path.join(CURRENT_DIR, "test_lab_orphans.json"),
        help="Where to write generated LAB orphan payload.",
    )
    parser.add_argument(
        "--matches-out",
        default=os.path.join(CURRENT_DIR, "test_lab_orphan_matches.json"),
        help="Where to write match results.",
    )
    args = parser.parse_args()

    tree = load_json_file(args.input)
    lab_node = find_child_folder(tree, "lab")
    if not lab_node:
        raise ValueError("Could not find 'lab' folder in input tree.")

    groups_data = load_json_file(args.groups)
    groups = [BackboneGroup(**g) for g in groups_data]

    orphans = build_lab_orphans(lab_node, args.db)
    save_json(orphans, args.orphans_out)
    print(f"Prepared {len(orphans)} LAB orphan items -> {args.orphans_out}")

    matches = run_direct_matching(args.backbone, groups, orphans)
    save_json(matches.model_dump(), args.matches_out)
    print(f"Matched {len(matches.matches)} assignments -> {args.matches_out}")


if __name__ == "__main__":
    main()
