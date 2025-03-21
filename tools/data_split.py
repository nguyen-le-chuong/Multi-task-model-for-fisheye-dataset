import os
import shutil
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process and split image segmentation dataset')
    

    parser.add_argument('--input_rgbLabels', type=str, required=True, 
                        help='Path to RGB labels directory')
    parser.add_argument('--input_rgbImages', type=str, required=True,
                        help='Path to RGB images directory')
    parser.add_argument('--input_2DBox', type=str, required=True,
                        help='Path to 2D box annotations directory')

    parser.add_argument('--output_segmentDir', type=str, default='segmented_classes',
                        help='Base directory for segmented class outputs')
    parser.add_argument('--output', type=str, default='woodscape_m',
                        help='Output directory for the final dataset')

    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
  
    parser.add_argument('--color_groups', type=str, default=None,
                        help='JSON string of color groups. If not provided, defaults will be used.')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel processing')
    parser.add_argument('--copy_files', action='store_true',
                        help='Copy files instead of moving them')
    
    return parser.parse_args()

def segment_image(image_file, input_folder, output_segmentDir, color_groups):
    """Process a single image for segmentation"""
    image_path = os.path.join(input_folder, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        for group_name, colors in color_groups.items():
            class_folder = os.path.join(output_segmentDir, group_name)
            os.makedirs(class_folder, exist_ok=True)

  
            segmented_image = np.zeros_like(image_array)
            for color in colors:
                color_mask = (image_array == color).all(axis=-1)
                segmented_image[color_mask] = color

            output_path = os.path.join(class_folder, image_file)
            Image.fromarray(segmented_image).save(output_path)
            
        return True
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")
        return False

def process_files(file_action, file_list, src_folder, dst_folder, extensions, desc):
    """Process (copy or move) files with a progress bar"""
    successful = 0
    with tqdm(total=len(file_list), desc=desc) as pbar:
        for base_name in file_list:
            for ext in extensions:
                src_path = os.path.join(src_folder, base_name + ext)
                dst_path = os.path.join(dst_folder, base_name + ext)
                if os.path.exists(src_path):
                    try:
                        file_action(src_path, dst_path)
                        successful += 1
                    except Exception as e:
                        print(f"Error processing {src_path}: {str(e)}")
            pbar.update(1)
    return successful

def main():
    args = parse_arguments()

    random.seed(args.seed)

    if args.color_groups is None:
        color_groups = {
            "person": [(255, 0, 0)],  
            "vehicle": [(0, 255, 255)],  
            "drivable": [(0, 0, 255), (255, 0, 255)],
            "curb": [(0, 255, 0)]
        }
    else:
        import json
        color_groups = json.loads(args.color_groups)

    os.makedirs(args.output_segmentDir, exist_ok=True)

    print("Step 1: Segmenting images by color groups...")
    input_folder = args.input_rgbLabels
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    with tqdm(total=len(image_files), desc="Segmenting images") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for image_file in image_files:
                future = executor.submit(
                    segment_image, 
                    image_file, 
                    input_folder, 
                    args.output_segmentDir, 
                    color_groups
                )
                futures.append(future)
                
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
    
    print(f"Segmentation completed. Results in '{args.output_segmentDir}'")

    output_root = args.output
    output_folders = {
        "images": os.path.join(output_root, "images"),
        "det_annotations": os.path.join(output_root, "det_annotations"),
        "drivable_area_annotations": os.path.join(output_root, "drivable_area_annotations"),
        "person_seg_annotations": os.path.join(output_root, "person_seg_annotations"),
        "vehicle_seg_annotations": os.path.join(output_root, "vehicle_seg_annotations"),
        "curb_seg_annotations": os.path.join(output_root, "curb_seg_annotations")
    }

    for key, path in output_folders.items():
        os.makedirs(os.path.join(path, "train"), exist_ok=True)
        os.makedirs(os.path.join(path, "val"), exist_ok=True)

    print("Step 3: Splitting dataset...")
    images_folder = args.input_rgbImages
    image_files = [f for f in os.listdir(images_folder) if f.endswith((".png"))]
    base_filenames = list(set(os.path.splitext(f)[0] for f in image_files))

    random.shuffle(base_filenames)
    split_idx = int(args.train_ratio * len(base_filenames))
    train_files = base_filenames[:split_idx]
    val_files = base_filenames[split_idx:]
    
    print(f"Dataset split: {len(train_files)} training files, {len(val_files)} validation files")

    print("Step 4: Distributing files to train/val directories...")

    file_action = shutil.copy if args.copy_files else shutil.move
    action_name = "Copying" if args.copy_files else "Moving"

    input_dirs = {
        "images": args.input_rgbImages,
        "det_annotations": args.input_2DBox,
        "drivable_area_annotations": os.path.join(args.output_segmentDir, "drivable"),
        "person_seg_annotations": os.path.join(args.output_segmentDir, "person"),
        "vehicle_seg_annotations": os.path.join(args.output_segmentDir, "vehicle"),
        "curb_seg_annotations": os.path.join(args.output_segmentDir, "curb")
    }
    

    for folder_key, src_folder in input_dirs.items():
        dst_folder = os.path.join(output_folders[folder_key], "train")
        ext = ".json" if folder_key == "det_annotations" else ".png"
        desc = f"{action_name} {folder_key} (train)"
        process_files(file_action, train_files, src_folder, dst_folder, [ext], desc)
    

    for folder_key, src_folder in input_dirs.items():
        dst_folder = os.path.join(output_folders[folder_key], "val")
        ext = ".json" if folder_key == "det_annotations" else ".png"
        desc = f"{action_name} {folder_key} (val)"
        process_files(file_action, val_files, src_folder, dst_folder, [ext], desc)

    print(f"Dataset preparation completed successfully! Output in '{args.output}'")

if __name__ == "__main__":
    main()