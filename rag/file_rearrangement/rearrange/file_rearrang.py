import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import List
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

def load_structure_file(file_path):
    """Load the markdown structure from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found at: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

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
    structure_file = os.path.join(base_dir, "study.md")
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
    main()
