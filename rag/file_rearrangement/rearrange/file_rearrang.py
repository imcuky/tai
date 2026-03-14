import os
import json
import sqlite3
import re
import argparse
from typing import List, Dict

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Pydantic Models
# =============================================================================

class FileDescription(BaseModel):
    relative_folder_path: str
    file: str
    description: str

class BackboneResult(BaseModel):
    backbone_path: str


class BackboneGroup(BaseModel):
    group_name: str
    main_item: str
    related_items: List[str]
    description: str


class BackboneGroupsResponse(BaseModel):
    groups: List[BackboneGroup]


class RearrangedGroup(BaseModel):
    group_name: str
    main_item: str
    related_items: List[str]


class OrphanMatch(BaseModel):
    item_path: str
    assigned_group: str
    #reason: str | None = None


class OrphanMatchResponse(BaseModel):
    matches: List[OrphanMatch]


class AggregationDecision(BaseModel):
    paths_to_aggregate: List[str]


# =============================================================================
# Helper Functions — Preprocessing
# =============================================================================

def load_json_file(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_debug_log(data: any, step_name: str, base_dir: str | None = None) -> str:
    """Save intermediate data to a JSON log file."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    file_path = os.path.join(log_dir, f"{step_name}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[DEBUG] Saved {step_name} log to: {file_path}")
    return file_path


def enrich_structure_with_descriptions(input_json_path: str, db_path: str, output_path: str) -> str:
    """Parse the input tree JSON (e.g. bfs_v3_tree.json), filter for 'study' category, 
    fetch descriptions from database, and generate a standardized list-based JSON tree."""
    print(f"Reading input structure from: {input_json_path}")

    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file not found: {input_json_path}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    files_statistics = {"processed": 0, "enriched": 0}

    def get_file_description(filename):
        cursor.execute(
            "SELECT description FROM file WHERE relative_path LIKE ? OR file_name = ? LIMIT 1",
            (f'%{filename}%', filename)
        )
        result = cursor.fetchone()
        return result[0].strip() if result and result[0] else None

    def process_node(node_data, node_name, current_path="", parent_force_keep=False):
        """
        Recursively process nodes.
        Logic:
        - Convert 'children' dict to list.
        - Merge 'files' dict into children list.
        - Filter: Keep node if category=='study' OR parent_force_keep is True OR it has kept children.
        """
        node_type = node_data.get("type", "folder")
        category = node_data.get("category", "")
        
        # If the category is "study", we identify this as a study root branch and keep everything under it.
        is_study = (category == "study")
        should_keep_this = is_study or parent_force_keep
        
        # Determine relative path
        rel_path = node_data.get("relative_path")
        # If relative path is missing in source, assume structure implies it (e.g. cleaning it up)
        # But usually we rely on existing paths. 
        if not rel_path:
             rel_path = f"{current_path}/{node_name}" if current_path else node_name
        
        # Replace root with study in relative path if it starts with root
        if rel_path.startswith("root"):
            rel_path = "study" + rel_path[4:]

        # --- Handle FILE ---
        if node_type == "file":
            filename = node_data.get("name", node_name)
            
            new_node = {
                "type": "file",
                "name": filename,
                "relative_path": rel_path
            }
            
            # Enrich
            desc = get_file_description(filename)
            if desc:
                new_node["description"] = desc
                files_statistics["enriched"] += 1
            
            files_statistics["processed"] += 1
            
            # If parent forced keep, we keep. 
            # If not forced, we only keep if this specific file is study? 
            # The prompt implies "material folder under category study". 
            # So if we are in a study folder, we keep.
            # If we are NOT in a study folder, do we keep this file?
            # Likely no, unless it's marked study itself.
            if should_keep_this:
                return new_node
            return None

        # --- Handle FOLDER ---
        new_node = {
            "type": "folder",
            "name": node_name,
            "relative_path": rel_path,
            "children": []
        }
        
        # Keep by_sequence for backbone identification
        by_sequence = node_data.get("by_sequence")
        if by_sequence is not None:
             new_node["by_sequence"] = by_sequence

        # 1. Process nested folders (children dict)
        children_dict = node_data.get("children", {})
        if isinstance(children_dict, dict):
            for child_name, child_data in children_dict.items():
                child_processed = process_node(child_data, child_name, rel_path, parent_force_keep=should_keep_this)
                if child_processed:
                    new_node["children"].append(child_processed)
        elif isinstance(children_dict, list):
             # Handle case where input might already be list-based (robustness)
             for child_data in children_dict:
                 cname = child_data.get("name", "unknown")
                 child_processed = process_node(child_data, cname, rel_path, parent_force_keep=should_keep_this)
                 if child_processed:
                    new_node["children"].append(child_processed)

        # 2. Process files (files dict)
        files_dict = node_data.get("files", {})
        if isinstance(files_dict, dict):
            for file_hash, file_data in files_dict.items():
                fname = file_data.get("name", "unknown")
                file_processed = process_node(file_data, fname, rel_path, parent_force_keep=should_keep_this)
                if file_processed:
                    new_node["children"].append(file_processed)

        # Decide whether to return this node
        if should_keep_this:
            return new_node
        elif len(new_node["children"]) > 0:
            # If we are just a container for study items, keep us
            return new_node
        else:
            return None

    # Start processing from root
    root_name = input_data.get("name", "root")
    enriched_root = process_node(input_data, root_name, "")
    
    conn.close()

    if enriched_root:
        # User requested root name to be 'study'
        enriched_root["name"] = "study"
        enriched_root["relative_path"] = "study"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_root, f, indent=2, ensure_ascii=False)
        print(f"Enriched JSON structure saved to: {output_path}")
        print(f"Processed {files_statistics['processed']} files, enriched {files_statistics['enriched']} with descriptions")
        return output_path
    else:
        print("Warning: No 'study' content found in input tree.")
        return ""


def extract_file_descriptions(enriched_data: Dict) -> List[FileDescription]:
    """Recursively walk the enriched JSON tree and extract file descriptions."""
    descriptions: List[FileDescription] = []

    def traverse(node: Dict, folder_path: str = ""):
        children = node.get('children', []) or []
        name = node.get('name', '')
        node_type = node.get('type', 'folder')

        if node_type == 'folder':
            current_path = f"{folder_path}/{name}" if folder_path else name
            for child in children:
                traverse(child, current_path)
        elif node_type == 'file':
            desc = node.get('description', '')
            if desc:
                descriptions.append(FileDescription(
                    relative_folder_path=folder_path,
                    file=name,
                    description=desc
                ))

    # Root node has no name, just children
    for child in enriched_data.get('children', []) or []:
        traverse(child)

    return descriptions


# =============================================================================
# Helper Functions — Orphan Collection & Matching
# =============================================================================

def aggregate_folder_descriptions(node: Dict, max_files: int = 5) -> str:
    """Recursively collect descriptions from child files to represent a folder."""
    descriptions = []
    own_desc = node.get('description', '').strip()
    if own_desc:
        descriptions.append(f"Folder: {own_desc}")

    count = 0
    # Use a stack for depth-first traversal to find file descriptions
    stack = [node]
    
    while stack and count < max_files:
        curr = stack.pop()
        children = curr.get('children', []) or []
        # Add children to stack (reverse order to process first child first if we want, or just append)
        for child in reversed(children):
            if child.get('type') == 'file':
                d = child.get('description', '').strip()
                n = child.get('name')
                if d and count < max_files:
                    descriptions.append(f"{n}: {d}")
                    count += 1
            elif child.get('type') == 'folder':
                stack.append(child)
                
    return " | ".join(descriptions)

def get_folder_candidates(enriched_data: Dict, backbone_path: str) -> List[Dict]:
    """Identify folders that are candidates for aggregation (orphans only)."""
    candidates = []
    backbone_path = backbone_path.rstrip('/')
    backbone_prefix = f"{backbone_path}/"
    
    def traverse(node: Dict, hierarchy_path: str = ""):
        name = node.get('name')
        children = node.get('children', []) or []
        
        # Root handling
        if not name:
            for child in children:
                traverse(child, hierarchy_path)
            return

        node_path = f"{hierarchy_path}/{name}" if hierarchy_path else name
        
        # Skip backbone and its children
        if node_path == backbone_path or node_path.startswith(backbone_prefix):
            return

        # Skip parents of the backbone (cannot aggregate a folder that contains the backbone)
        if backbone_path.startswith(f"{node_path}/"):
             # Recurse but do NOT add as candidate
             for child in children:
                traverse(child, node_path)
             return

        node_type = node.get('type', 'folder')
        
        if node_type == 'folder':
            file_children = [c.get('name') for c in children if c.get('type') == 'file']
            folder_children = [c.get('name') for c in children if c.get('type') == 'folder']
            
            # Heuristic: Only consider folders with no subfolders for aggregation.
            # If a folder has subfolders, it is likely a container and should be traversed, not aggregated.
            has_subfolders = len(folder_children) > 0

            # Add to candidates only if it has files and NO subfolders
            if file_children and not has_subfolders:
                candidates.append({
                    "path": node_path,
                    "num_files": len(file_children),
                    "num_subfolders": len(folder_children),
                    "sample_files": file_children[:5],
                    "sample_subfolders": folder_children[:5],
                    "folder_description": aggregate_folder_descriptions(node)
                })
            
            # Recurse
            for child in children:
                traverse(child, node_path)

    for child in enriched_data.get('children', []) or []:
        traverse(child)
        
    return candidates


def run_aggregation_analysis(enriched_data: Dict, backbone_path: str) -> List[str]:
    """Ask LLM which folders should be aggregated into single units."""
    candidates = get_folder_candidates(enriched_data, backbone_path)
    if not candidates:
        return []

    save_debug_log(candidates, "02_0_aggregation_candidates")
    print(f"Analyzing {len(candidates)} folders for potential aggregation...")

    client = OpenAI()
    
    # Process in batches if too many candidates, but usually folder count is manageable
    # For robust handling, let's limit context or batch if needed.
    # Here assuming < 100 folders, fits in context easily.
    
    completion = client.beta.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intelligent course file organizer.\n"
                    "You will analyze a list of folders from a course directory (excluding the main lecture backbone).\n"
                    "For each folder, determine if it should be treated as a **single unified item** (Aggregated)"
                    "or if its contents should be split and treated individually based on the individual files content and structure inside of the folder.\n\n"
                    "Rules for should 'Aggregate':\n"
                    "- The folder represents a cohesive SINGLE unit (e.g., 'Project 1', 'Lab 3').\n"
                    "- It contains multiple files that belong together (e.g., instructions, starter code, solution) and breaking them apart loses context.\n"
                    "- Subfolders are structural (e.g., 'src', 'tests', 'lib') and NOT independent content units.\n\n"

                    "Rules for 'Split' (Do NOT Aggregate):\n"
                    "- The folder is named a general container (not specific like 'Project 1'/Discussion XX) for organizational purposes but the files inside are independent (e.g., 'Slides' folder with individual lecture PDFs that can be treated separately).\n"
                    # "- The folder contains contains independent sub-units topics (e.g., 'File on AI', 'File on Mathematics') that is clearly does not related to each other.\n"         
                    # "- A 'mixed' folder with both files and subfolders should usually be SPLIT to preserve the subfolder hierarchy."

                    "NOTE: Unless it's very clear that the files are not tightly related to the parent folder and the subfolders are just organizational, it's safer to NOT split.\n\n"
                    "DO NOT ADD FILES THAT DOES NOT EXIST IN THE INPUT."
                    "\nReturn a JSON object with a list 'paths_to_aggregate' containing the paths of folders to KEEP TOGETHER."
                )
            },
            {
                "role": "user",
                "content": json.dumps(candidates, indent=2)
            }
        ],
        response_format=AggregationDecision,
        seed=42
    )
    
    decision = completion.choices[0].message.parsed
    print(f"Aggregation decision: Keeping {len(decision.paths_to_aggregate)} folders as units.")
    save_debug_log(decision.model_dump(), "02_0_aggregation_decision")
    return decision.paths_to_aggregate


def collect_orphan_items(enriched_data: Dict, backbone_path: str, aggregated_paths: List[str] = None) -> List[Dict]:
    """Collect items that are NOT in the backbone folder."""
    orphans: List[Dict] = []
    aggregated_paths = set(aggregated_paths or [])
    
    # Ensure no trailing slash for consistent comparison
    backbone_folder = backbone_path.rstrip('/')
    
    # We'll use this prefix to robustly detect children of the backbone
    backbone_prefix = f"{backbone_folder}/"

    def traverse(node: Dict, hierarchy_path: str = ""):
        name = node.get('name')
        node_type = node.get('type', 'folder')
        children = node.get('children', []) or []

        # Root case: just traverse children (root has no name)
        if not name:
            for child in children:
                traverse(child, hierarchy_path)
            return

        node_path = f"{hierarchy_path}/{name}" if hierarchy_path else name

        # 1. Exact match: The node IS the backbone folder -> Skip it and all its children
        if node_path == backbone_folder:
            return
        
        # 2. Child match: The node is INSIDE the backbone folder -> Skip it
        if node_path.startswith(backbone_prefix):
            return

        description = node.get('description', '')
        
        # New Aggregation Heuristic: Folders containing only files are treated as units
        if node_type == 'folder':
             # Only aggregate if it has files and NO subfolders
             # (And ensure we aren't accidentally aggregating the backbone parent if structured weirdly)
             folder_children = [c for c in children if c.get('type') == 'folder']
             file_children = [c for c in children if c.get('type') == 'file']

             should_aggregate = (len(folder_children) == 0 and len(file_children) > 0)
             
             if should_aggregate:
                  # Compute description from files
                  combined_desc = f"Folder containing: {', '.join([c.get('name') for c in file_children])}"
                  # Try to get richer descriptions if available
                  rich_descs = [f"{c.get('name')}: {c.get('description', '')}" for c in file_children if c.get('description')]
                  if rich_descs:
                       combined_desc += ". Details: " + " | ".join(rich_descs)
                  
                  orphans.append({
                        'structure_path': node_path,
                        'relative_path': node.get('relative_path', node_path),
                        'name': name,
                        'type': 'folder_unit', # Mark as a unit so matcher knows it's a folder
                        'description': combined_desc
                  })
                  return # Stop recursing into children files

        # Aggregation Check (Legacy / Manual Override):
        # If this folder is marked for aggregation, treat it as a file/unit and stop recursing.
        if node_type == 'folder' and node_path in aggregated_paths:
             # Ensure we do NOT aggregate the parent folder of the backbone
             if not backbone_path.startswith(f"{node_path}/"):
                 orphans.append({
                    'structure_path': node_path,
                    'relative_path': node.get('relative_path', node_path),
                    'name': name,
                    'type': 'folder (aggregated)',
                    'description': aggregate_folder_descriptions(node)
                })
                 return # Do not recurse info children

        # Standard Recursion Logic:
        if node_type == 'folder':
            if not children:
                return  # Skip empty folders

            for child in children:
                traverse(child, node_path)
            return

        # File Logic:
        # If we reach this point, it's a file that wasn't inside a "leaf folder" unit.
        # This occurs if the parent folder was "structural" (had subfolders) but also contained loose files.
        orphans.append({
            'structure_path': node_path,
            'relative_path': node.get('relative_path', node_path),
            'name': name,
            'type': node_type,
            'description': description
        })

    for child in enriched_data.get('children', []) or []:
        traverse(child)

    return orphans


def build_summary(
    items: List[any],
    limit: int = None,
    truncate_fields: Dict[str, int] = None
) -> List[Dict]:
    """
    Dynamically build a summary list from Pydantic models or dictionaries.
    
    Args:
        items: List of Pydantic models, dicts, or other objects.
        limit: Max number of items to process (None for all).
        truncate_fields: Dict mapping field names to max length (e.g., {'description': 200}).
    """
    summary = []
    # If items is None, return empty list
    if not items:
        return []
        
    slice_end = limit if limit is not None else len(items)
    items_slice = items[:slice_end]

    for item in items_slice:
        data = {}
        # Convert to dict
        if isinstance(item, BaseModel):
            data = item.model_dump()
        elif isinstance(item, dict):
            # Shallow copy to avoid modifying original
            data = item.copy()
        else:
            # Skip unsupported types
            continue

        # Post-process fields (truncation)
        if truncate_fields:
            for field, max_len in truncate_fields.items():
                val = data.get(field)
                if isinstance(val, str) and len(val) > max_len:
                    data[field] = val[:max_len] + "..."
        
        summary.append(data)

    return summary


def extract_backbone_subtree(enriched_data: Dict, backbone_path: str) -> Dict | None:
    """Find and return the backbone folder node from the enriched tree."""
    backbone_path = backbone_path.rstrip('/')

    def find(node: Dict, hierarchy_path: str = "") -> Dict | None:
        name = node.get('name', '')
        children = node.get('children', []) or []

        node_path = f"{hierarchy_path}/{name}" if hierarchy_path else name

        if not name:
            for child in children:
                result = find(child, hierarchy_path)
                if result:
                    return result
            return None

        if node_path == backbone_path:
            return node

        if backbone_path.startswith(node_path):
            for child in children:
                result = find(child, node_path)
                if result:
                    return result

        return None

    for child in enriched_data.get('children', []) or []:
        result = find(child)
        if result:
            return result
    return None


def merge_matches_into_groups(
    groups: List[BackboneGroup], matches: OrphanMatchResponse
) -> List[RearrangedGroup]:
    """Merge orphan matches into backbone groups, producing RearrangedGroups."""
    # This function now calls our new helper to generate the complete plan
    # but returns RearrangedGroup objects to fit typed return signature if needed elsewhere
    plan_dicts = generate_rearrangement_plan(groups, matches)
    
    # Convert dicts back to RearrangedGroup objects for compatibility
    return [
        RearrangedGroup(
            group_name=p["group_name"],
            main_item=p["main_item"],
            related_items=p["related_items"]
        )
        for p in plan_dicts
    ]


def generate_rearrangement_plan(
    backbone_groups: List[BackboneGroup],
    matches: OrphanMatchResponse
) -> List[Dict]:
    """
    Combine backbone groups and orphan matches into a final rearrangement plan.
    Returns a list of dictionaries representing the new structure.
    """
    # 1. Initialize groups with backbone main items
    # Use lowercase keys for robust matching, but preserve original casing for display
    plan_map = {}
    
    def normalize_key(name):
        return name.strip().lower()

    # Add existing backbone groups
    for bg in backbone_groups:
        key = normalize_key(bg.group_name)
        
        # Initialize if not exists (should be unique from backbone generation, but safe to check)
        if key not in plan_map:
            # Use getattr with default to be safe, though Pydantic should ensure fields exist
            initial_related = getattr(bg, 'related_items', [])
            if initial_related is None:
                initial_related = []
            
            plan_map[key] = {
                "group_name": bg.group_name,
                "main_item": bg.main_item,
                "description": bg.description,
                "related_items": list(initial_related)  # Copy the list
            }
        else:
            # If duplicate group name exists, merge related items
            existing_entry = plan_map[key]
            
            # Merge related items
            new_related = getattr(bg, 'related_items', []) or []
            for item in new_related:
                if item not in existing_entry["related_items"]:
                    existing_entry["related_items"].append(item)
            
            # Update description if current is missing and new one has it
            if bg.description:
                if existing_entry["description"] and existing_entry["description"] != bg.description:
                    existing_entry["description"] += f" | {bg.description}"
                elif not existing_entry["description"]:
                    existing_entry["description"] = bg.description
            
            # Update main_item if current is missing and new one has it
            if not existing_entry["main_item"] and bg.main_item:
                existing_entry["main_item"] = bg.main_item

    # 2. Distribute orphans into groups
    for match in matches.matches:
        raw_target_group = match.assigned_group.strip()
        orphan_path = match.item_path.strip()

        # Handle "New:" prefix for dynamically created groups
        target_group_display = raw_target_group
        if target_group_display.lower().startswith("new:"):
            # extract the part after 'New:'
            target_group_display = target_group_display.split(":", 1)[1].strip()

        key = normalize_key(target_group_display)

        # Create new group if it doesn't exist
        if key not in plan_map:
            plan_map[key] = {
                "group_name": target_group_display,
                "main_item": None,  # New groups might not have a backbone anchor
                "description": "Dynamically created group",
                "related_items": []
            }

        # Add orphan to the group
        current_related = plan_map[key]["related_items"]
        if orphan_path not in current_related:
            current_related.append(orphan_path)

    # 3. Convert dicts to RearrangedGroup objects
    final_plan_objs = []
    
    for p in plan_map.values():
        main_item = p.get("main_item")
        if main_item is None:
            main_item = ""
            
        final_plan_objs.append({
            "group_name": p["group_name"],
            "main_item": main_item,
            "related_items": p["related_items"],
            "description": p.get("description", "")
        })
    
    # Save plan to debug log as a clean list of dicts
    save_debug_log(final_plan_objs, "06_rearrangement_plan")
    
    print(f"Generated rearrangement plan with {len(final_plan_objs)} groups.")
    return final_plan_objs


# =============================================================================
# Pipeline Function 1: Backbone Identification
# =============================================================================

def run_backbone_identification(enriched_data: Dict) -> str:
    """Identify the main backbone folder from the enriched structure.

    1. Extract file descriptions from enriched data.
    2. Ask LLM to identify the chronological backbone folder.
    3. Return the backbone path.
    """
    file_descriptions = extract_file_descriptions(enriched_data)

    descriptions_payload = [fd.model_dump() for fd in file_descriptions]
    save_debug_log(descriptions_payload, "01_backbone_descriptions_payload")

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intelligent file system organizer for university course materials.\n"
                    "Given the study folder structure and each file description, "
                    "identify the 'Main Type' folder that best serves as the chronological "
                    "backbone of the course.\n"
                    "The backbone should be the folder containing core lecture materials "
                    "that provides the main chronological structure for the course (etc. Lecture Slides).\n"
                    "Return only the relative folder path of the backbone."
                )
            },
            {
                "role": "user",
                "content": json.dumps(descriptions_payload, indent=2)
            }
        ],
        response_format=BackboneResult,
        seed=42
    )

    result: BackboneResult = completion.choices[0].message.parsed
    print(f"Identified backbone: {result.backbone_path}")
    save_debug_log(result.backbone_path, "01_backbone_path")
    return result.backbone_path


# =============================================================================
# Pipeline Function 2: Orphan Matching (Plan Matching)
# =============================================================================

def run_plan_matching(enriched_data: Dict, backbone_path: str) -> OrphanMatchResponse:
    """Match non-backbone items to backbone groups.

    Step A: Generate backbone groups from backbone subfolders via LLM.
    Step B: Collect orphans and match each to a group via LLM.
    """
    client = OpenAI()

    # --- Step A: Generate backbone groups ---
    backbone_subtree = extract_backbone_subtree(enriched_data, backbone_path)
    if not backbone_subtree:
        raise ValueError(f"Backbone folder '{backbone_path}' not found in enriched data.")

    backbone_json = json.dumps(backbone_subtree, indent=2)
    save_debug_log(backbone_subtree, "02_1_backbone_subtree")

    groups_completion = client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are organizing university course materials.\n"
                    "Given the backbone folder structure (the main lecture folder), "
                    "generate a rearranged for main backbone folder into a structure logical     group for sub-unit.\n"
                    "For material file that are related, create a group "
                    "with a descriptive name (e.g., group_name: 'Lecture xx: <Topic>'), "
                    "as main_item, and a brief description of the topic.\n"
                    "Only create groups from the Backbone folder subfolder — these represent "
                    "the chronological units of the course.\n"
                    "At the end, also include 'Lecture Miscellaneous' group with no main_itemfor files that don't fit into a specific lecture unit.\n"
                    "CRITICAL RULE: You MUST use exact file paths from the input JSON for 'main_item' and 'related_items'. Do NOT invent or guess file paths (e.g. do not add 'sol-disc12' if it is not in the input)."
                )
            },
            {
                "role": "user",
                "content": backbone_json
            }
        ],
        response_format=BackboneGroupsResponse,
        seed=42
    )

    backbone_groups: List[BackboneGroup] = groups_completion.choices[0].message.parsed.groups
    print(f"Generated {len(backbone_groups)} backbone groups.")
    save_debug_log([g.model_dump() for g in backbone_groups], "02_2_backbone_groups")

    # --- Step B: Determine Aggregation Strategy ---
    print("Step Extra: Running aggregation analysis for folders...")
    # aggregated_paths = run_aggregation_analysis(enriched_data, backbone_path)
    aggregated_paths = []
    
    # --- Step C: Collect orphans and match ---
    orphans = collect_orphan_items(enriched_data, backbone_path, aggregated_paths)
    print(f"Identified {len(orphans)} orphan items needing placement.")
    save_debug_log(orphans, "02_3_orphans_collected")

    if not orphans:
        print("No orphans detected.")
        return OrphanMatchResponse(matches=[])
    
    # Print some sample orphans to verify collection
    print("Sample orphans:")
    for o in orphans[:5]:
        print(f" - {o['name']} ({o['type']})")

    # Convert backbone_groups to dicts so they are JSON serializable
    # and truncate long descriptions if necessary
    groups_summary = build_summary(backbone_groups)
    
    # Process orphans (they are already dicts)
    # We pass None for limit here if we want ALL orphans processed
    # However, to avoid token limits, we should process in batches.
    #all_orphans = build_summary(orphans, limit=None, truncate_fields={'description': 200})
    
    save_debug_log(groups_summary, "03_groups_for_matching")
    save_debug_log(orphans, "04_orphans")
    #save_debug_log(all_orphans, "05_orphan_summary")

    all_matches = []
    chunk_size = 50  # Adjust based on average token size per item
    
    total_orphans = len(orphans)
    print(f"Processing {total_orphans} orphans in batches of {chunk_size}...")

    for i in range(0, total_orphans, chunk_size):
        batch_orphans = orphans[i:i + chunk_size]
        print(f"Processing batch {i // chunk_size + 1} / {(total_orphans + chunk_size - 1) // chunk_size}...")
        
        try:
            match_completion = client.beta.chat.completions.parse(
                model="gpt-5-mini", 
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an intelligent course material organizer for any subject (Computer Science, Math, Literature, etc.).\n\n"
                            f"The folder '{backbone_path}' defines the chronological 'backbone' of this course.\n"
                            f"You will receive:\n"
                            f"- A list of 'Existing Groups' (the backbone units) with their descriptions.\n"
                            f"- A batch of 'Orphan Files' that need to be categorized.\n\n"

                            f"Your Task:\n"
                            f"For EACH orphan, assign it to the most semantically relevant group. If you think this material can have multiple matches, assign it to the most relevant one.\n\n"

                            f"Matching Considerations:\n"
                            f"1. **Strong Match (Preferred)**: If the file's name or description strongly relates to a specific backbone unit's topic/descriptions.\n"
                            f"   - Example: A file named 'Derivatives Practice' or on Derivatives fits into 'Lecture XX: Differentiation' or Derivatives topic is include in the group description.\n"
                            
                            f"2. **Ambiguous/No Match (Fallback)**: If the file does not clearly fit any specific backbone unit, place it in 'Lecture Miscellaneous' category\n"

                            f"NOTE: Try to infer its topic from its name. If the Orphan description and the name is not informative, it's safer to put it in Miscellaneous than to risk misplacement.\n"
                            
                            # f"   Create a new group by prefixing with 'New: '.\n"
                            # f"   - Examples: 'New: Exams', 'New: Homework', 'New: Resources', 'New: Miscellaneous'. NOTE: Can you rearrange previous assigned file to new group if it's more appropriate\n\n"
                            f"Constraints:\n"
                            f"- Use existing 'group_name' exactly as provided when matching.\n"
                            # f"- Every single orphan file must appear in the output list exactly once.\n"
                            f"- Do NOT create files that do not exist in orphans.\n"
                            # f"- Return structured JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "backbone_folder": backbone_path,
                            "existing_groups": groups_summary,
                            "orphans": batch_orphans
                        }, indent=2)
                    }
                ],
                response_format=OrphanMatchResponse,
                seed=42
            )
            
            batch_result = match_completion.choices[0].message.parsed
            if batch_result and batch_result.matches:
                # Post-processing: Filter out hallucinations (items not in the input)
                # Ensure we match against the 'relative_path' or 'structure_path' sent to the model
                # The model sees 'relative_path' as the key if it was in the input dump.
                valid_orphan_paths = set()
                for o in batch_orphans:
                    valid_orphan_paths.add(o.get('relative_path'))
                    valid_orphan_paths.add(o.get('structure_path')) # Just in case model returns structure path
                
                filtered_matches = []
                
                for match in batch_result.matches:
                    # Clean up path just in case
                    cleaned_path = match.item_path.strip()
                    
                    if cleaned_path in valid_orphan_paths:
                        filtered_matches.append(match)
                    else:
                        print(f"  - Warning: Filtered out hallucinated item: {cleaned_path}")

                all_matches.extend(filtered_matches)
                print(f"  - Matched {len(filtered_matches)} valid items in this batch.")
            else:
                print(f"  - Warning: No matches returned for this batch.")

        except Exception as e:
            print(f"  - Error processing batch: {e}")
            # Continue to next batch instead of crashing entirely

    matches = OrphanMatchResponse(matches=all_matches)
    print(f"Matched total of {len(matches.matches)} orphan items to groups.")
    merge_matches_into_groups(backbone_groups, matches)
    return matches


# =============================================================================
# Pipeline Function 3: File Rearrangement (stub)
# =============================================================================

def run_file_rearrangement(orphan_matches: OrphanMatchResponse, enriched_data: Dict):
    """Placeholder for file rearrangement — not implemented yet."""
    raise NotImplementedError("File rearrangement is not yet implemented.")


# =============================================================================
# Utility: Build Structure Tree from Plan
# =============================================================================

def load_file_hashes(db_path: str) -> Dict[str, str]:
    """Load all file hashes from the database into a dictionary keyed by relative_path."""
    if not os.path.exists(db_path):
        print(f"Warning: Database not found at {db_path}. Hashes will be empty.")
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT relative_path, file_hash, file_name FROM file")
        rows = cursor.fetchall()
        result = {}
        for row in rows:
            path, h, filename = row
            if not path: continue
            
            # 1. Exact path
            result[path] = h
            
            # 2. Strip "CS 61A/" prefix
            if path.startswith("CS 61A/"):
                stripped = path[7:]
                result[stripped] = h
                
            # 3. Filename based (fallback, might overwrite duplicates but useful)
            if filename:
                result[f"__NAME__{filename}"] = h
                
        print(f"Loaded {len(rows)} file rows from database. Lookup map size: {len(result)}")
        return result
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}
    finally:
        conn.close()


def index_enriched_data(enriched_data: Dict) -> Dict[str, Dict]:
    """Index the enriched data tree by relative_path AND name for fast lookups.
    Returns: { 'path/to/file': node, '__NAME__file.ext': node }
    """
    index = {}
    
    def traverse(node):
        path = node.get('relative_path')
        name = node.get('name')
        
        if path:
            index[path] = node
        if name:
            # Prefix with __NAME__ to avoid collisions with relative paths that might just look like names
            index[f"__NAME__{name}"] = node
        
        for child in node.get('children', []) or []:
            traverse(child)
    
    for child in enriched_data.get('children', []) or []:
        traverse(child)
        
    return index


def build_node_recursive(original_node: Dict, hash_map: Dict[str, str]) -> Dict:
    """Recursively build a node for the new tree, populating hashes for files."""
    new_node = {
        "type": original_node.get("type", "folder"),
        "name": original_node.get("name"),
        "relative_path": original_node.get("relative_path"),
        "description": original_node.get("description", "")
    }
    
    if new_node["type"] == "file":
        rel_path = new_node.get("relative_path", "")
        # 1. Try exact path match
        file_hash = hash_map.get(rel_path)
        
        if not file_hash:
            # 2. Try filename fallback
            fname = new_node.get("name")
            if fname:
                file_hash = hash_map.get(f"__NAME__{fname}")
        
        new_node["file_hash"] = file_hash or ""
        
    elif new_node["type"] == "folder":
        new_node["children"] = []
        for child in original_node.get("children", []) or []:
            new_node["children"].append(build_node_recursive(child, hash_map))
            
    return new_node


def build_rearranged_structure_tree(plan_path: str, enriched_data_path: str, db_path: str, output_path: str):
    """
    Builds a full hierarchical tree based on the rearrangement plan, expanding folders
    and adding file hashes from the database.
    """
    print("Building rearranged structure tree...")
    
    # 1. Load Data
    try:
        plan_data = load_json_file(plan_path)
        enriched_data = load_json_file(enriched_data_path)
        hash_map = load_file_hashes(db_path)
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return

    # 2. Index Enriched Data
    enriched_index = index_enriched_data(enriched_data)
    
    # 3. Build Tree
    result_tree = []
    
    # plan_data is expected to be a list of groups
    if isinstance(plan_data, dict) and "groups" in plan_data:
        plan_groups = plan_data["groups"]
    elif isinstance(plan_data, list):
        plan_groups = plan_data
    else:
        print("Error: Invalid plan format")
        return

    for group in plan_groups:
        group_name = group.get("group_name", "Unnamed Group")
        # main_item might be empty string or None
        main_item = group.get("main_item")
        related_items = group.get("related_items", [])
        
        group_node = {
            "type": "group",
            "name": group_name,
            "children": []
        }
        
        items_to_process = []
        if main_item:
            items_to_process.append(main_item)
        if related_items:
            items_to_process.extend(related_items)
            
        # Deduplicate while preserving order, just in case
        seen = set()
        unique_items = []
        for x in items_to_process:
            if x and x not in seen:
                unique_items.append(x)
                seen.add(x)
        
        for item_path in unique_items:
            # Find in index
            original_node = enriched_index.get(item_path)
            
            # Fallback for filenames
            if not original_node:
                basename = os.path.basename(item_path)
                original_node = enriched_index.get(f"__NAME__{basename}")

            if original_node:
                # Build recursive copy with hashes
                new_item_node = build_node_recursive(original_node, hash_map)
                group_node["children"].append(new_item_node)
            else:
                try:
                    print(f"Warning: Item not found in enriched data: {item_path}")
                except UnicodeEncodeError:
                    print(f"Warning: Item not found in enriched data: {item_path.encode('ascii', 'replace').decode('ascii')}")
                
                # Create a placeholder if not found
                basename = os.path.basename(item_path)
                
                # Try finding a fallback hash
                fallback_hash = hash_map.get(item_path, "")
                if not fallback_hash:
                    # Also try finding hash by basename in the map (requires load_file_hashes to support __NAME__ keys)
                    fallback_hash = hash_map.get(f"__NAME__{basename}", "")
                
                group_node["children"].append({
                    "type": "unknown",
                    "relative_path": item_path,
                    "name": basename,
                    "file_hash": fallback_hash,
                })
        
        result_tree.append(group_node)
        
    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_tree, f, indent=2, ensure_ascii=False)
        
    print(f"Rearranged structure tree saved to: {output_path}")


# =============================================================================
# CLI Orchestration
# =============================================================================

def run_enrichment(base_dir: str, input_filename: str) -> str:
    """Preprocessing: parse input tree JSON + SQLite DB → enriched JSON."""
    workspace_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    
    # Input now comes from 'input' folder
    input_file = os.path.join(base_dir, "input", input_filename)
        
    db_file = os.path.join(workspace_root, "cs61a_metadata.db")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    enriched_output = os.path.join(output_dir, "study_enriched.json")

    print("=" * 60)
    print("Preprocessing: Enriching structure with file descriptions")
    print("=" * 60)
    print(f"Database path: {db_file}")
    print(f"Input file path: {input_file}")
    
    return enrich_structure_with_descriptions(input_file, db_file, enriched_output)


def _load_enriched(base_dir: str) -> Dict:
    enriched_file = os.path.join(base_dir, "outputs", "study_enriched.json")
    if not os.path.exists(enriched_file):
        # Fallback to root-level file from previous runs
        enriched_file = os.path.join(base_dir, "study_enriched.json")
    return load_json_file(enriched_file)


def _test_orphan_collection(base_dir: str):
    enriched_data = _load_enriched(base_dir)
    backbone_file = os.path.join(base_dir, "outputs", "backbone_result.json")
    backbone_path = load_json_file(backbone_file)["backbone_path"]
    orphans = collect_orphan_items(enriched_data, backbone_path)
    save_debug_log(orphans, "test_orphans_collected")
    print(f"Collected {len(orphans)} orphans:")
    for o in orphans:
        print(f"- {o['structure_path']} (type: {o['type']})")

def _run_pipeline(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if args.step in ("enrich", "all"):
            # If running all, ensure input is provided or default used
            input_file = args.input
            run_enrichment(base_dir, input_file)

        if args.step in ("backbone", "all"):
            print("=" * 60)
            print("Step 1: Backbone Identification")
            print("=" * 60)
            enriched_data = _load_enriched(base_dir)
            backbone_path = run_backbone_identification(enriched_data)

            backbone_output = os.path.join(output_dir, "backbone_result.json")
            with open(backbone_output, 'w', encoding='utf-8') as f:
                json.dump({"backbone_path": backbone_path}, f, indent=4)
            print(f"Backbone result saved to: {backbone_output}")

        if args.step in ("match", "all"):
            print("=" * 60)
            print("Step 2: Orphan Matching")
            print("=" * 60)
            enriched_data = _load_enriched(base_dir)

            # Load backbone from previous step or run it
            backbone_file = os.path.join(output_dir, "backbone_result.json")
            if os.path.exists(backbone_file):
                backbone_path = load_json_file(backbone_file)["backbone_path"]
            else:
                print("No backbone result found, running backbone identification first...")
                backbone_path = run_backbone_identification(enriched_data)

            matches = run_plan_matching(enriched_data, backbone_path)

            # Save raw matches
            matches_output = os.path.join(output_dir, "orphan_matches.json")
            with open(matches_output, 'w', encoding='utf-8') as f:
                json.dump(matches.model_dump(), f, indent=4)
            print(f"Orphan matches saved to: {matches_output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Course folder rearrangement pipeline")
    parser.add_argument(
        "--step",
        choices=["enrich", "backbone", "match", "all", "tree"],
        default="all",
        help="Pipeline step: 'enrich', 'backbone', 'match', 'all', or 'tree'."
    )
    parser.add_argument(
        "--input",
        required=False,
        default="bfs_v3_tree.json",
        help="Input JSON filename located in 'input' folder (e.g. bfs_v3_tree.json)."
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if args.step == "tree":
        # Run only the tree building step
        plan_path = os.path.join(base_dir, "logs", "06_rearrangement_plan.json")
        enriched_path = os.path.join(base_dir, "outputs", "study_enriched.json")
        if not os.path.exists(enriched_path):
             enriched_path = os.path.join(base_dir, "study_enriched.json")
        
        # Determine DB path
        workspace_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
        db_path = os.path.join(workspace_root, "cs61a_metadata.db")
        
        output_tree_path = os.path.join(output_dir, "rearrangement_structure_tree.json")
        
        build_rearranged_structure_tree(plan_path, enriched_path, db_path, output_tree_path)
    else:
        _run_pipeline(args)

    #