import json
import sys

def check_json_structure(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Top-level keys:", list(data.keys()))
    
    for key in data.keys():
        print(f"\nExamining '{key}':")
        if isinstance(data[key], list):
            print(f"  '{key}' is a list with {len(data[key])} items")
            if data[key]:
                first_item = data[key][0]
                print("  First item keys:", list(first_item.keys()))
                if 'hidden_states' in first_item:
                    print("  'hidden_states' in first item is a list with", len(first_item['hidden_states']), "items")
                    print("  First hidden_state keys:", list(first_item['hidden_states'][0].keys()))
        else:
            print(f"  '{key}' is not a list")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_json_structure.py <path_to_json_file>")
        sys.exit(1)
    
    check_json_structure(sys.argv[1])