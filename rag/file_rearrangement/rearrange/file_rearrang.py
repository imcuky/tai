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


class BackboneFolderSelection(BaseModel):
    """Model for LLM to identify the main backbone folder"""
    main_backbone_folder: str
    relative_path: str


class OrphanMatch(BaseModel):
    item_path: str
    assigned_group: str
    


class OrphanMatchResponse(BaseModel):
    matches: List[OrphanMatch]

def load_structure_file(file_path):
    """Load the markdown structure from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found at: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def build_hierarchical_structure_from_db(db_path, output_path, course_prefix="CS 61A"):
    """
    Build a hierarchical JSON structure directly from the database.
    The relative_path in the database already contains the organization structure.
    ONLY processes files in the 'study' folder.
    
    Args:
        db_path: Path to the SQLite database (e.g., CS 61A_metadata.db)
        output_path: Path to save the enriched JSON structure
        course_prefix: Prefix to strip from relative paths (default: "CS 61A")
    """
    print(f"Building hierarchical structure from database: {db_path}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch all files with their information - ONLY from study folder
    query = """
        SELECT file_name, relative_path, description, url
        FROM file
        WHERE relative_path IS NOT NULL 
          AND relative_path != ''
          AND (relative_path LIKE '%/study/%' OR relative_path LIKE 'study/%')
        ORDER BY relative_path
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} files in study folder")
    
    # Build hierarchical structure
    root = {'children': []}
    
    for file_name, relative_path, description, url in rows:
        # Strip course prefix if present
        if relative_path.startswith(f"{course_prefix}/"):
            path = relative_path[len(course_prefix) + 1:]
        else:
            path = relative_path
        
        # Split path into parts
        parts = path.split('/')
        
        # Navigate/create folder structure
        current = root
        for i, part in enumerate(parts[:-1]):  # All parts except the last (filename)
            # Look for existing folder
            folder = None
            for child in current.get('children', []):
                if child.get('name') == part and child.get('type') == 'folder':
                    folder = child
                    break
            
            # Create folder if not exists
            if not folder:
                folder_path = '/'.join(parts[:i+1])
                folder = {
                    'type': 'folder',
                    'name': part,
                    'relative_path': folder_path,
                    'children': []
                }
                if 'children' not in current:
                    current['children'] = []
                current['children'].append(folder)
            
            current = folder
        
        # Add file node
        file_node = {
            'type': 'file',
            'name': file_name,
            'relative_path': relative_path
        }
        if description:
            file_node['description'] = description
        if url:
            file_node['url'] = url
        
        if 'children' not in current:
            current['children'] = []
        current['children'].append(file_node)
    
    # Write JSON to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(root, f, indent=2, ensure_ascii=False)
    
    print(f"Hierarchical structure saved to: {output_path}")
    print(f"Processed {len(rows)} files")
    
    return output_path


def enrich_structure_with_descriptions(structure_md_path, db_path, output_path):
    """
    DEPRECATED: This function is kept for backward compatibility.
    Use build_hierarchical_structure_from_db() instead.
    
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


def get_folder_structure_summary(enriched_data: Dict, max_depth: int = 3) -> str:
    """
    Generate a summary of the folder structure for LLM analysis.
    
    Args:
        enriched_data: Hierarchical JSON structure
        max_depth: Maximum depth to traverse (default: 3)
    
    Returns:
        String representation of folder structure
    """
    lines = []
    
    def traverse(node: Dict, depth: int = 0, prefix: str = ""):
        if depth > max_depth:
            return
        
        name = node.get('name', '')
        node_type = node.get('type', 'folder')
        children = node.get('children', []) or []
        
        if not name and depth == 0:
            # Root node
            for child in children:
                traverse(child, depth, prefix)
            return
        
        indent = "  " * depth
        if node_type == 'folder':
            # Count files and subfolders
            file_count = sum(1 for c in children if c.get('type') == 'file')
            folder_count = sum(1 for c in children if c.get('type') == 'folder')
            
            path = node.get('relative_path', name)
            info = f"{indent}📁 {name}/ ({file_count} files, {folder_count} folders)"
            lines.append(info)
            
            # Traverse children
            for child in children:
                traverse(child, depth + 1, f"{prefix}{name}/")
        else:
            # File node - just count, don't list all files
            pass
    
    for child in enriched_data.get('children', []):
        traverse(child, 0)
    
    return '\n'.join(lines)


def identify_main_backbone_folder(enriched_data: Dict) -> str:
    """
    Use LLM to identify the main backbone folder that should serve as the
    chronological backbone of the course.
    
    Args:
        enriched_data: Hierarchical JSON structure
    
    Returns:
        Path to the main backbone folder (e.g., "study/lecture")
    """
    client = OpenAI()
    
    # Generate a summary of the folder structure
    structure_summary = get_folder_structure_summary(enriched_data, max_depth=3)
    
    system_prompt = """
You are analyzing a course material directory structure to identify the "main backbone folder".

The main backbone folder should:
1. Contain the primary chronological teaching materials (typically lecture slides, lecture notes, or similar)
2. Be organized sequentially (e.g., lecture01, lecture02, etc.)
3. Serve as the natural organizing structure for other course materials

Common patterns:
- "study/lecture" or "lecture" folders often contain the main lecture materials
- Look for folders with sequential numbering (lec01, lec02, etc.)
- Lecture materials typically include slides, code examples, and notes

Your task: 
1. Identify which folder path should be the main backbone
2. Fill in 'main_backbone_folder' with a descriptive name (e.g., "lecture")
3. Fill in 'relative_path' with the FULL folder path from the structure (e.g., "study/lecture")

IMPORTANT: The 'relative_path' must be the complete path to the folder, not just the folder name.
Example: If the backbone is the lecture folder under study, relative_path should be "study/lecture", NOT "study/" or "lecture/".
"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Here is the course folder structure:\n\n{structure_summary}\n\nIdentify the main backbone folder path."
        }
    ]
    
    completion = client.beta.chat.completions.parse(
        model="gpt-5-nano",
        messages=messages,
        response_format=BackboneFolderSelection
    )
    
    result = completion.choices[0].message.parsed
    print(f"Main backbone folder name: {result.main_backbone_folder}")
    print(f"Main backbone relative path: {result.relative_path}")
    
    return result.relative_path


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

def generate_rearrangement_plan(enriched_data: Dict, backbone_folder: str) -> RearrangementPlan:
    """
    Generate a rearrangement plan based on the backbone folder.
    Groups files from the backbone folder into logical lecture/topic groups.
    
    Args:
        enriched_data: Hierarchical JSON structure
        backbone_folder: Path to the main backbone folder (e.g., "study/lecture")
    
    Returns:
        RearrangementPlan with grouped structure
    """
    client = OpenAI() 
    
    # Extract files and folders from the backbone folder
    backbone_items = []
    
    def extract_backbone_items(node: Dict, hierarchy_path: str = ""):
        name = node.get('name', '')
        node_type = node.get('type', 'folder')
        children = node.get('children', []) or []
        description = node.get('description', '')
        
        if not name and not hierarchy_path:
            # Root node
            for child in children:
                extract_backbone_items(child, "")
            return
        
        node_path = f"{hierarchy_path}/{name}" if hierarchy_path else name
        
        # Check if this is within or is the backbone folder
        if node_path == backbone_folder:
            # We're at the backbone root, extract its children
            for child in children:
                extract_backbone_items(child, node_path)
            return
        
        if not node_path.startswith(f"{backbone_folder}/"):
            # Not in backbone folder, check children for backbone
            if backbone_folder.startswith(node_path):
                for child in children:
                    extract_backbone_items(child, node_path)
            return
        
        # This item is within the backbone folder
        if node_type == 'file':
            backbone_items.append({
                'path': node_path,
                'name': name,
                'type': node_type,
                'description': description[:200] if description else ''
            })
        elif node_type == 'folder':
            # Check if this is a leaf folder (contains only files)
            has_subfolders = any(c.get('type') == 'folder' for c in children)
            if not has_subfolders and children:
                # Leaf folder with files
                backbone_items.append({
                    'path': node_path,
                    'name': name,
                    'type': 'folder',
                    'file_count': len(children),
                    'description': f"Folder containing {len(children)} files"
                })
            else:
                # Recurse into subfolder
                for child in children:
                    extract_backbone_items(child, node_path)
    
    for child in enriched_data.get('children', []):
        extract_backbone_items(child, "")
    
    print(f"Found {len(backbone_items)} items in backbone folder: {backbone_folder}")
    
    system_prompt = f"""
You are organizing course materials for a university course.

The backbone folder "{backbone_folder}" contains the main chronological teaching materials (lectures/topics).

Your task:
1. Analyze the items in the backbone folder
2. Group them into logical lecture/topic units
3. Each group should have:
   - A descriptive name (e.g., "Lecture 01: Introduction to Python")
   - A main item (typically the main slide or primary material)
   - Related items (other materials for that topic)

Grouping Logic:
- Look for sequential patterns (lec01, lec02, lecture01, etc.)
- Group materials by topic or lecture number
- Main item should be the primary teaching material (usually slides with "_1pp.pdf" or similar)
- Related items are supporting materials (code examples, additional slides, etc.)
- Keep groups focused on single topics/lectures
- Preserve the chronological order of the course

Example:
- study/lecture/lec01/slides/01-Welcome_1pp.pdf → Main item for "Lecture 01: Welcome"
- study/lecture/lec01/slides/01.py → Related item for "Lecture 01: Welcome"
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
                "items": backbone_items[:200]  # Limit to avoid context overflow
            }, indent=2)
        }
    ]
    
    completion = client.beta.chat.completions.parse(
        model="gpt-5-mini-2025-08-07",
        messages=messages,
        response_format=RearrangementPlan
    )
    
    return completion.choices[0].message.parsed

def run_enrichment_from_db(base_dir: str, db_path: str = None):
    """
    Step 1: Build hierarchical structure directly from database (NEW APPROACH).
    ONLY processes files in the 'study' folder.
    
    Args:
        base_dir: Base directory for output files
        db_path: Path to database (if None, will look in 'new database' subfolder)
    """
    print("=" * 60)
    print("Step 1: Building hierarchical structure from database (study folder only)")
    print("=" * 60)
    
    if db_path is None:
        # Default to 'new database' subfolder
        db_path = os.path.join(base_dir, "new database", "CS 61A_metadata.db")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    enriched_output = os.path.join(base_dir, "study_enriched.json")
    
    print(f"Database path: {db_path}")
    build_hierarchical_structure_from_db(db_path, enriched_output, course_prefix="CS 61A")
    print(f"step 1 completed: {enriched_output}")


def run_backbone_identification(base_dir: str):
    """
    Step 2: Identify the main backbone folder using LLM (NEW APPROACH).
    
    Args:
        base_dir: Base directory containing study_enriched.json
    
    Returns:
        Path to the main backbone folder
    """
    print("\n" + "=" * 60)
    print("Step 2: Identifying main backbone folder")
    print("=" * 60)
    
    enriched_file = os.path.join(base_dir, "study_enriched.json")
    enriched_data = load_json_file(enriched_file)
    
    backbone_folder = identify_main_backbone_folder(enriched_data)
    print(f"step 2 completed: Main backbone folder is '{backbone_folder}'")
    
    return backbone_folder


def run_plan_generation(base_dir: str, backbone_folder: str = None):
    """
    Step 3: Generate rearrangement plan based on backbone folder (NEW APPROACH).
    
    Args:
        base_dir: Base directory containing study_enriched.json
        backbone_folder: Path to main backbone folder (if None, will auto-identify)
    """
    print("\n" + "=" * 60)
    print("Step 3: Generating rearrangement plan")
    print("=" * 60)
    
    enriched_file = os.path.join(base_dir, "study_enriched.json")
    output_file = os.path.join(base_dir, "rearrangement_plan_study.json")
    
    enriched_data = load_json_file(enriched_file)
    
    # If backbone folder not provided, identify it
    if backbone_folder is None:
        backbone_folder = identify_main_backbone_folder(enriched_data)
    
    print(f"Using backbone folder: {backbone_folder}")
    print("Generating rearrangement plan...")
    
    plan_object = generate_rearrangement_plan(enriched_data, backbone_folder)
    parsed_plan = plan_object.model_dump()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_plan, f, indent=4)
    
    print(f"step 3 completed: {output_file}")
    print(f"Created {len(parsed_plan.get('rearranged_structure', []))} lecture groups.")


def run_enrichment(base_dir: str):
    """
    DEPRECATED: Old approach using markdown file.
    Use run_enrichment_from_db() instead.
    """
    workspace_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    structure_file = os.path.join(base_dir, "study.md")
    db_file = os.path.join(workspace_root, "cs61a_metadata.db")
    enriched_output = os.path.join(base_dir, "study_enriched.json")

    print("=" * 60)
    print("Step 1: Enriching structure with file descriptions")
    print("=" * 60)
    print(f"Database path: {db_file}")
    enrich_structure_with_descriptions(structure_file, db_file, enriched_output)


def main():
    """
    DEPRECATED: Old main function.
    Use the new workflow functions instead:
    1. run_enrichment_from_db()    2. run_backbone_identification()    3. run_plan_generation()
    4. run_orphan_matching()
    """
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
        enriched_data = load_json_file(os.path.join(base_dir, "study_enriched.json"))
        backbone_folder = identify_main_backbone_folder(enriched_data)
        plan_object = generate_rearrangement_plan(enriched_data, backbone_folder)
        
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
    parser = argparse.ArgumentParser(
        description="CS 61A study folder rearrangement helper (NEW WORKFLOW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NEW WORKFLOW (Database-based):
  Default: Run all steps sequentially (no arguments needed)
  
  Optional arguments for running individual steps:
    --step enrich     → Build study_enriched.json from database
    --step identify   → Identify main backbone folder using LLM
    --step plan       → Generate rearrangement plan based on backbone
    --step match      → Match orphan items to lecture groups
        """
    )
    parser.add_argument(
        "--step",
        choices=["enrich", "identify", "plan", "match"],
        default=None,
        help="Optional: Choose specific step to run (default: run full workflow)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to database file (default: ./new database/CS 61A_metadata.db)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Manually specify backbone folder path (skips identification step)"
    )
    
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # If no step specified, run full workflow
        if args.step is None:
            # Run full workflow
            print("\n" + "="*60)
            print("RUNNING FULL WORKFLOW")
            print("="*60 + "\n")
            
            # Step 1: Build enriched structure from database
            run_enrichment_from_db(base_dir, args.db)
            
            # Step 2: Identify backbone folder (or use provided one)
            if args.backbone:
                backbone = args.backbone
                print(f"\nUsing provided backbone folder: {backbone}")
            else:
                backbone = run_backbone_identification(base_dir)
            
            # Step 3: Generate rearrangement plan
            run_plan_generation(base_dir, backbone)
            
            # Step 4: Match orphan items
            run_orphan_matching(base_dir)
            
            print("\n" + "="*60)
            print("✓ FULL WORKFLOW COMPLETED")
            print("="*60)
            print("\nGenerated files:")
            print(f"  - study_enriched.json")
            print(f"  - rearrangement_plan_study.json")
            print(f"  - rearrangement_plan_study_final.json")
        
        # Individual step execution    
        elif args.step == "enrich":
            run_enrichment_from_db(base_dir, args.db)
            
        elif args.step == "identify":
            backbone = run_backbone_identification(base_dir)
            print(f"\n{'='*60}")
            print(f"Backbone folder: {backbone}")
            print(f"{'='*60}")
            
        elif args.step == "plan":
            run_plan_generation(base_dir, args.backbone)
            
        elif args.step == "match":
            run_orphan_matching(base_dir)
            
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()

