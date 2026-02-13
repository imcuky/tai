import os
import json
import sqlite3
import re
import argparse
from typing import List, Dict, Set

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pydantic Models for Structured Output
class RearrangedGroup(BaseModel):
    group_name: str
    main_item: str
    related_items: List[str]

class RearrangementPlan(BaseModel):
    main_backbone_folder: str
    rearranged_structure: List[RearrangedGroup]


class OrphanMatch(BaseModel):
    item_path: str
    assigned_group: str
    reason: str | None = None


class OrphanMatchResponse(BaseModel):
    matches: List[OrphanMatch]

def load_structure_file(file_path):
    """Load the markdown structure from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found at: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def enrich_structure_with_descriptions(structure_md_path, db_path, output_path):
    """
    Parse the markdown structure file, fetch descriptions from database,
    and generate a JSON file with hierarchical structure and descriptions.
    
    Args:
        structure_md_path: Path to the input structure markdown file (e.g., study.md)
        db_path: Path to the SQLite database (e.g., cs61a_metadata.db)
        output_path: Path to save the enriched JSON structure
    """
    print(f"Reading structure from: {structure_md_path}")
    
    # Read the markdown structure
    with open(structure_md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Connect to database
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Build hierarchical structure
    root = {}
    stack = [(root, -1, "")]  # (current_dict, indent_level, current_path)
    files_processed = 0
    files_with_descriptions = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Calculate indentation level
        indent_match = re.match(r'^(\s*)', line)
        indent_level = len(indent_match.group(1)) if indent_match else 0
        
        # Check if it's a directory line (bold markdown with slashes)
        dir_match = re.search(r'\*\*(.+?)/\*\*', stripped)
        if dir_match:
            dir_name = dir_match.group(1)
            
            # Pop stack to correct level
            while stack and stack[-1][1] >= indent_level:
                stack.pop()
            
            # Create new directory entry
            parent_node, _, parent_path = stack[-1]
            if 'children' not in parent_node:
                parent_node['children'] = []
            
            # Construct relative path for folder
            if parent_path:
                folder_rel_path = f"{parent_path}/{dir_name}"
            else:
                folder_rel_path = dir_name
            
            new_dir = {
                'type': 'folder',
                'name': dir_name,
                'relative_path': folder_rel_path,
                'children': []
            }
            parent_node['children'].append(new_dir)
            stack.append((new_dir, indent_level, folder_rel_path))
            continue
        
        # Check if it's a file line
        file_match = re.search(r'^\s*-\s+([^\*]+)$', stripped)
        if file_match:
            filename = file_match.group(1).strip()
            
            # Skip .json and .yaml files
            if filename.endswith('.json') or filename.endswith('.yaml'):
                continue
            
            # Skip if it's actually a folder indicator
            if filename.endswith('/') or '**' in filename:
                continue
            
            files_processed += 1
            
            # Pop stack to correct level
            while stack and stack[-1][1] >= indent_level:
                stack.pop()
            
            parent = stack[-1][0]
            if 'children' not in parent:
                parent['children'] = []
            
            # Query database for description
            query = """
                SELECT description, relative_path 
                FROM file 
                WHERE relative_path LIKE ? OR file_name = ?
                LIMIT 1
            """
            cursor.execute(query, (f'%{filename}%', filename))
            result = cursor.fetchone()
            
            file_entry = {
                'type': 'file',
                'name': filename
            }
            
            if result:
                if result[0]:
                    description = result[0].strip()
                    files_with_descriptions += 1
                    file_entry['description'] = description
                if result[1]:
                    file_entry['relative_path'] = result[1].strip()
            
            parent['children'].append(file_entry)
    
    conn.close()
    
    # Write JSON to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(root, f, indent=2, ensure_ascii=False)
    
    print(f"Enriched JSON structure saved to: {output_path}")
    print(f"Processed {files_processed} files")
    print(f"Found descriptions for {files_with_descriptions} files")
    
    return output_path


def load_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_existing_plan_paths(plan_data: Dict) -> Set[str]:
    paths: Set[str] = set()
    for group in plan_data.get('rearranged_structure', []):
        main_item = group.get('main_item')
        if main_item:
            paths.add(main_item)
        for item in group.get('related_items', []) or []:
            if item:
                paths.add(item)
    return paths


def collect_orphan_items(enriched_data: Dict, backbone_folder: str, existing_paths: Set[str]) -> List[Dict]:
    """Return nodes living under the backbone but not yet part of the plan."""
    orphans: List[Dict] = []
    backbone_folder = backbone_folder.rstrip('/')
    slides_prefix = f"{backbone_folder}/assets/slides"

    def should_traverse(h_path: str) -> bool:
        return backbone_folder.startswith(h_path) or h_path.startswith(backbone_folder)

    def traverse(node: Dict, hierarchy_path: str = ""):
        name = node.get('name')
        node_type = node.get('type', 'folder')
        children = node.get('children', []) or []

        if not name:
            for child in children:
                traverse(child, hierarchy_path)
            return

        node_path = f"{hierarchy_path}/{name}" if hierarchy_path else name

        if not should_traverse(node_path):
            return

        # Always explore until we hit the backbone root
        if node_path == backbone_folder:
            for child in children:
                traverse(child, node_path)
            return

        if node_path.startswith(slides_prefix):
            return

        # Already accounted for in plan? skip
        if node_path in existing_paths:
            # still dive deeper in case nested nodes missing
            for child in children:
                traverse(child, node_path)
            return

        # Candidate evaluation (only consider descendants of backbone)
        if node_path.startswith(f"{backbone_folder}/"):
            rel_path = node.get('relative_path', node_path)
            description = node.get('description', '')

            if node_type == 'folder' and children:
                child_types = {child.get('type', 'file') for child in children}
                if 'folder' in child_types:
                    # keep digging until we hit final-level folders/files
                    for child in children:
                        traverse(child, node_path)
                    return
                # Folder only contains files -> treat folder as unit
                orphans.append({
                    'structure_path': node_path,
                    'relative_path': rel_path,
                    'name': name,
                    'type': node_type,
                    'description': description
                })
                return

            # File node -> add as orphan
            orphans.append({
                'structure_path': node_path,
                'relative_path': rel_path,
                'name': name,
                'type': node_type,
                'description': description
            })
            return

        # Recurse deeper for any other cases leading toward backbone
        for child in children:
            traverse(child, node_path)

    for child in enriched_data.get('children', []) or []:
        traverse(child)

    return orphans


def build_group_summary(groups: List[Dict]) -> List[Dict]:
    summary = []
    for group in groups:
        summary.append({
            'name': group.get('group_name'),
            'main_item': group.get('main_item'),
            'count': len(group.get('related_items', []) or [])
        })
    return summary


def build_orphan_summary(orphans: List[Dict], limit: int = 200) -> List[Dict]:
    summary = []
    for orphan in orphans[:limit]:
        summary.append({
            'path': orphan['structure_path'],
            'relative_path': orphan['relative_path'],
            'type': orphan['type'],
            'description': (orphan.get('description') or '')[:200]
        })
    return summary


def match_orphans_to_groups(orphans: List[Dict], plan_data: Dict, backbone_folder: str) -> OrphanMatchResponse:
    client = OpenAI()

    groups_summary = build_group_summary(plan_data.get('rearranged_structure', []))
    orphan_summary = build_orphan_summary(orphans)

    system_prompt = f"""
You are reorganizing course files for CS 61A.

The lecture slides inside {backbone_folder} already define the backbone. You will receive:
- Existing groups (lectures) with their main slide.
- Orphan folders/files that live under the same study folder but are not yet assigned.

For EACH orphan item:
1. Assign it to the most relevant existing group (use group name exactly).
   - Match by numbering, topic words, or description keywords.
   - Example: study/lecture/disc/disc01 -> Lecture 01: Welcome.
2. Only if no existing group fits, create a new group with label "New: <Descriptive Name>".
   - Examples: "New: Practice Exams", "New: Discussion Solutions".
3. Keep assignments concise; every orphan must appear exactly once.
4. Return structured JSON per the response schema.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": json.dumps({
                "backbone_folder": backbone_folder,
                "existing_groups": groups_summary,
                "orphans": orphan_summary
            }, indent=2)
        }
    ]

    completion = client.beta.chat.completions.parse(
        model="gpt-5-mini-2025-08-07",
        messages=messages,
        response_format=OrphanMatchResponse
    )

    return completion.choices[0].message.parsed


def merge_matches_into_plan(plan_data: Dict, matches: OrphanMatchResponse) -> Dict:
    groups = plan_data.get('rearranged_structure', [])
    group_map: Dict[str, Dict] = {g['group_name']: g for g in groups}

    for match in matches.matches:
        path = match.item_path.strip()
        target_group = match.assigned_group.strip()

        if not path:
            continue

        if target_group.lower().startswith('new:'):
            new_name = target_group.split(':', 1)[1].strip() or 'New Category'
            if new_name not in group_map:
                group_map[new_name] = {
                    'group_name': new_name,
                    'main_item': '',
                    'related_items': []
                }
                groups.append(group_map[new_name])
            target_group = new_name

        if target_group not in group_map:
            group_map[target_group] = {
                'group_name': target_group,
                'main_item': '',
                'related_items': []
            }
            groups.append(group_map[target_group])

        related_items = group_map[target_group].setdefault('related_items', [])
        if path not in related_items:
            related_items.append(path)

    return plan_data


def run_orphan_matching(base_dir: str):
    enriched_file = os.path.join(base_dir, "study_enriched.json")
    plan_file = os.path.join(base_dir, "rearrangement_plan_study.json")
    output_file = os.path.join(base_dir, "rearrangement_plan_study_final.json")

    print("=" * 60)
    print("Step 2: Matching non-backbone folders to backbone groups")
    print("=" * 60)

    enriched_data = load_json_file(enriched_file)
    plan_data = load_json_file(plan_file)
    backbone = plan_data.get('main_backbone_folder', 'study/lecture')

    existing_paths = collect_existing_plan_paths(plan_data)
    orphans = collect_orphan_items(enriched_data, backbone, existing_paths)

    print(f"Identified {len(orphans)} orphan items needing placement.")
    if not orphans:
        print("No orphans detected; plan unchanged.")
        return

    matches = match_orphans_to_groups(orphans, plan_data, backbone)
    updated_plan = merge_matches_into_plan(plan_data, matches)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_plan, f, indent=4)

    print(f"Updated plan saved to: {output_file}")


def run_enrichment(base_dir: str):
    workspace_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    structure_file = os.path.join(base_dir, "study.md")
    db_file = os.path.join(workspace_root, "cs61a_metadata.db")
    enriched_output = os.path.join(base_dir, "study_enriched.json")

    print("=" * 60)
    print("Step 1: Enriching structure with file descriptions")
    print("=" * 60)
    print(f"Database path: {db_file}")
    enrich_structure_with_descriptions(structure_file, db_file, enriched_output)

def generate_rearrangement_plan(structure_content):
    """Ask LLM to identify the main type and reorganize files into groups."""
    client = OpenAI() 
    
    system_prompt = """
    You are an intelligent file system organizer for university course materials.
    
    Goal: 
    1. Analyze the provided directory tree (Markdown).
    2. Identify the "Main Type" folder that best serves as the chronological backbone of the course.
    3. Create a new logical structure where files from other folders are grouped into the corresponding "Topic" unit defined by the backbone.
    
    Matching Logic:
    - Match items primarily by content/topic relevance. For example, if "Lecture 1" covers "Introduction to AI", then other materials that relate to that topic should be grouped with it.
    - Use string similarity/topic matching if numbers are missing or ambiguous.
    - If a file doesn't clearly match a lecture, try to find the best fit based on content clues or place it in a "Miscellaneous" group under the backbone.
    - The backbone should ideally be the folder that contains the core lecture materials, as it provides the main structure for the course. Other folders should be organized around this backbone.
    - Just match lowest folder level items instead of the entire items in the folder to a single backbone folder.
    - If you want to move a folder that contains multiple items (e.g., "practice/hw/sol-hw01/hw01.py", "practice/hw/sol-hw01/hw01.py_metadata.yaml"), simply listing the parent folder (e.g., "practice/hw/sol-hw01") in 'related_items' is sufficient; this implies moving every item under that folder.
    - If a file doesn't fit well, place it in a "Miscellaneous" group under the backbone.
    """
    
    completion = client.beta.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the course file structure:\n\n{structure_content}"}
        ],
        response_format=RearrangementPlan
    )
    
    return completion.choices[0].message.parsed

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    structure_file = os.path.join(base_dir, "study_enriched.md")
    output_file = os.path.join(base_dir, "rearrangement_plan_study.json")
    
    print(f"Reading structure from: {structure_file}")
    
    try:
        content = load_structure_file(structure_file)
        
        # Verify content isn't too large for context (truncate if necessary, though 128k is large)
        # if len(content) > 400000:
        #     print("Warning: Structure file is large, sending first 400000 chars...")
        #     content = content[:400000]
            
        print("Sending to LLM to generate rearrangement plan...")
        plan_object = generate_rearrangement_plan(content)
        
        # Convert Pydantic model to dict for saving
        parsed_plan = plan_object.model_dump()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_plan, f, indent=4)
            
        print(f"Success! Rearrangement plan saved to: {output_file}")
        print(f"Selected Backbone: {parsed_plan.get('main_backbone_folder')}")
        print(f"Created {len(parsed_plan.get('rearranged_structure', []))} groups.")

        # Execute rearrangement
        # execute_rearrangement(parsed_plan, base_dir)
        
    except Exception as e:
        print(f"Error: {str(e)}")

def execute_rearrangement(plan_data, base_dir):
    """Move files into the new directory structure based on the plan."""
    import shutil

    # Create root for rearranged files to avoid messing up original structure immediately
    # or we can move them in place. Let's create a 'sorted_course_materials' folder.
    output_root = os.path.join(base_dir, "sorted_course_materials")
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")

    # The paths in the plan are relative to the structure.md location or absolute? 
    # Usually LLM extracts paths as seen in markdown. The markdown shows a tree structure.
    # We need to find the actual source files.
    # Assuming 'rag/file_rearrangement/rearrange' is where we are, and valid files are nearby?
    # Context check: The structure.md often reflects 'rag/file_rearrangement/lecture/...' etc.
    # We need to locate the source root. 
    # Based on the user's workspace, 'rearrange' is a subfolder. The actual files seem to be in 'rag/file_rearrangement'.
    
    # We'll assume source_root is the parent of the script's directory for now, 
    # or we search for the files.
    source_root = os.path.dirname(base_dir) # Go up one level from 'rearrange' to 'file_rearrangement'

    for group in plan_data.get('rearranged_structure', []):
        group_name = group['group_name']
        group_folder = os.path.join(output_root, group_name.replace(":", " -")) # Sanitize name
        
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        # Collect all items to move
        items = [group['main_item']] + group['related_items']
        
        for item_path in items:
            # item_path from LLM usually looks like "lecture/01-Welcome.pdf"
            # We need to find this file in source_root
            
            # Clean path from markdown artifacts if any
            clean_path = item_path.strip().replace('*', '') 
            
            # Construct absolute source path
            # Warning: LLM might return just filenames or partial paths. 
            # We strictly assume the paths match the structure.md hierarchy.
            src_full_path = os.path.join(source_root, clean_path)
            
            if os.path.exists(src_full_path):
                # Copy instead of move for safety during testing
                dst_full_path = os.path.join(group_folder, os.path.basename(clean_path))
                try:
                    if os.path.isdir(src_full_path):
                         shutil.copytree(src_full_path, dst_full_path, dirs_exist_ok=True)
                         print(f"Copied directory: {clean_path} -> {group_name}")
                    else:
                        shutil.copy2(src_full_path, dst_full_path)
                        # print(f"Copied file: {clean_path} -> {group_name}")
                except Exception as file_err:
                    print(f"Failed to copy {clean_path}: {file_err}")
            else:
                print(f"Source file not found: {clean_path} (checked: {src_full_path})")

    print(f"\nRearrangement complete. Files copied to: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 61A study folder rearrangement helper")
    parser.add_argument(
        "--step",
        choices=["enrich", "match", "plan"],
        default="enrich",
        help="Choose 'enrich' to build study_enriched.json, 'plan' to regenerate the backbone plan, or 'match' to attach non-backbone folders to lecture groups."
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # try:
    #     if args.step == "enrich":
    #         run_enrichment(base_dir)
    #     elif args.step == "plan":
    #         main()
    #     else:
    #         run_orphan_matching(base_dir)
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    run_orphan_matching(base_dir)
