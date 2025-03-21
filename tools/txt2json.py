import os
import json
import argparse
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert text annotation files to JSON format.')
    parser.add_argument('--input', '-i', type=str, default="/mnt/HDD/chuong/temp/YOLOP/data/box_2d_annotations",
                        help='Input directory containing text annotation files')
    parser.add_argument('--output', '-o', type=str, default="/mnt/HDD/chuong/temp/YOLOP/data/box_2d_json_annotations",
                        help='Output directory for JSON files')
    parser.add_argument('--category-map', '-c', type=str, default=None,
                        help='Path to JSON file containing category mapping (optional)')
    parser.add_argument('--timestamp', '-t', type=int, default=10000,
                        help='Timestamp value for frames (default: 10000)')
    return parser.parse_args()

def load_category_map(map_file=None):
    """Load category mapping from file or use default."""
    if map_file and os.path.exists(map_file):
        with open(map_file, 'r') as f:
            return json.load(f)
    else:
        # Default mapping
        return {
            "vehicles": "car",
            "person": "pedestrian",
            "traffic_light": "traffic sign"
        }

def convert_files(input_folder, output_folder, category_map, timestamp):
    """Convert text annotation files to JSON format."""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all txt files
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    # Process each file with progress bar
    for filename in tqdm(txt_files, desc="Converting files"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename.replace(".txt", ".json"))
        
        objects = []
        try:
            with open(input_file_path, "r") as file:
                for line_num, line in enumerate(file, 1):
                    parts = line.strip().split(",")
                    if len(parts) != 6:
                        print(f"Warning: Skipping invalid line {line_num} in {filename}: {line.strip()}")
                        continue
                        
                    category, obj_id, x1, y1, x2, y2 = parts
                    
                    try:
                        obj_id = int(obj_id)
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    except ValueError as e:
                        print(f"Error in {filename}, line {line_num}: {e}")
                        continue
                    
                    # Map categories to required JSON format
                    category = category_map.get(category, category)
                    
                    # Create object entry
                    obj_entry = {
                        "category": category,
                        "id": obj_id,
                        "attributes": {
                            "occluded": False,
                            "truncated": False,
                            "trafficLightColor": "none"
                        },
                        "box2d": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    }
                    objects.append(obj_entry)
            
            # Construct final JSON structure
            json_data = {
                "name": filename.replace(".txt", ""),
                "frames": [
                    {
                        "timestamp": timestamp,
                        "objects": objects
                    }
                ]
            }
            
            # Save JSON file
            with open(output_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():
    """Main function to orchestrate the conversion process."""
    args = parse_arguments()
    category_map = load_category_map(args.category_map)
    
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Using timestamp: {args.timestamp}")
    
    convert_files(args.input, args.output, category_map, args.timestamp)
    print(f"\nConversion complete! JSON files saved to: {args.output}")

if __name__ == "__main__":
    main()