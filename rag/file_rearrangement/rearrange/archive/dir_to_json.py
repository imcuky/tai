import os
import json

def folder_to_dict(path, root_dir=None):
    """
    Recursively traverse a directory and build a dictionary representation of it.
    """
    if root_dir is None:
        root_dir = os.path.dirname(path)
        
    name = os.path.basename(path)
    relative_path = os.path.relpath(path, root_dir).replace('\\', '/')
    
    node = {
        "name": name,
        "relative_path": relative_path,
        "type": "folder" if os.path.isdir(path) else "file"
    }
    
    if os.path.isdir(path):
        node["children"] = []
        try:
            for item in os.listdir(path):
                node["children"].append(folder_to_dict(os.path.join(path, item), root_dir))
        except PermissionError:
            pass # Skip directories we don't have access to
            
    return node

def convert_study_folder_to_json(input_folder, output_json_path):
    """
    Converts a folder structure into a JSON file.
    """
    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return
        
    structure = folder_to_dict(input_folder)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully converted folder structure to JSON at: {output_json_path}")

if __name__ == "__main__":
    # Pointing to the specific folder mentioned
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input", "study_106b")
    output_json = os.path.join(base_dir, "input", "bfs_v3_tree_study_106b.json")
    
    convert_study_folder_to_json(input_folder, output_json)
