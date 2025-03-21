import os
import cv2
import json
import numpy as np
import argparse
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def extract_lanes_from_mask(mask, num_points=72, cluster_eps=20, match_thresh=60):
    """
    Extract lane coordinates from a binary mask image.
    
    Args:
        mask (numpy.ndarray): Binary mask image where lane pixels are non-zero
        num_points (int): Number of points to sample along the y-axis
        cluster_eps (int): DBSCAN epsilon parameter for clustering lane points
        match_thresh (int): Maximum distance threshold for matching lanes between rows
        
    Returns:
        tuple: (formatted_lanes, h_samples)
            - formatted_lanes: List of lanes, each containing x-coordinates (-2 for missing points)
            - h_samples: List of y-coordinates where lanes were sampled
    """
    H, W = mask.shape
    h_samples = np.linspace(0.04 * H, 0.86 * H, num_points, dtype=int).tolist()
    lanes = [[] for _ in range(num_points)]
    active_lanes = []  

    for i, y in enumerate(h_samples):
        x_positions = np.where(mask[y, :] > 0)[0]
        if len(x_positions) > 0:
            x_positions = x_positions.reshape(-1, 1)
            clustering = DBSCAN(eps=cluster_eps, min_samples=2).fit(x_positions)
            lane_clusters = []

            for label in np.unique(clustering.labels_):
                if label == -1:  # Skip noise points
                    continue
                cluster = x_positions[clustering.labels_ == label].flatten()
                avg_x = int(np.mean(cluster))
                lane_clusters.append(avg_x)

            lane_clusters.sort()

            if not active_lanes:  # First row with detections
                active_lanes = [[x] for x in lane_clusters]
            else:
                new_active_lanes = [[] for _ in range(len(active_lanes))]
                used = set()
                
                for x in lane_clusters:
                    best_idx = -1
                    min_dist = match_thresh  
                    for j, lane in enumerate(active_lanes):
                        if lane and j not in used:
                            last_x = lane[-1]
                            dist = abs(last_x - x)
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = j

                    if best_idx != -1:
                        new_active_lanes[best_idx] = active_lanes[best_idx] + [x]
                        used.add(best_idx)
                    else:
                        new_active_lanes.append([x])  # Create new lane

                active_lanes = new_active_lanes

            lanes[i] = [lane[-1] if lane else -2 for lane in active_lanes]  
        else:
            lanes[i] = [-2] * len(active_lanes) 

    num_lanes = max(len(lane) for lane in lanes) if lanes else 0
    formatted_lanes = [[-2] * num_points for _ in range(num_lanes)]

    for i, lane_xs in enumerate(lanes):
        for j, x in enumerate(lane_xs): 
            if j < num_lanes:
                formatted_lanes[j][i] = x  
    
    if num_lanes == 0:
        formatted_lanes = [[-2] * num_points]
        
    return formatted_lanes, h_samples

def generate_tusimple_annotations(mask_dir, image_dir, output_json, num_points=72, cluster_eps=20, match_thresh=60):
    """
    Generate TuSimple format annotations from a directory of mask images.
    
    Args:
        mask_dir (str): Directory containing lane segmentation masks
        image_dir (str): Directory containing corresponding original images
        output_json (str): Path to output JSON file
        num_points (int): Number of points to sample for each lane
        cluster_eps (int): DBSCAN epsilon parameter for clustering
        match_thresh (int): Maximum distance for matching lanes between rows
    """
    annotations = []
    
    mask_files = [f for f in sorted(os.listdir(mask_dir)) if f.endswith(('.png', '.jpg'))]
    
    for mask_filename in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, mask_filename)
        image_path = os.path.join(image_dir, mask_filename)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
        if mask is None:
            print(f"Warning: Failed to read mask: {mask_path}")
            continue
            
        lanes, h_samples = extract_lanes_from_mask(
            mask, 
            num_points=num_points, 
            cluster_eps=cluster_eps, 
            match_thresh=match_thresh
        )
        annotation = {
            "lanes": lanes,
            "h_samples": h_samples,
            "raw_file": image_path
        }
        annotations.append(annotation)

    with open(output_json, 'w') as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + '\n')
    
    print(f"Generated {len(annotations)} annotations saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(description='Generate TuSimple format lane annotations from segmentation masks')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing lane segmentation masks')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing corresponding original images')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--num_points', type=int, default=72, help='Number of points to sample for each lane')
    parser.add_argument('--cluster_eps', type=int, default=20, help='DBSCAN epsilon parameter for clustering')
    parser.add_argument('--match_thresh', type=int, default=60, help='Maximum distance for matching lanes between rows')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mask_dir):
        raise ValueError(f"Mask directory does not exist: {args.mask_dir}")
    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generate_tusimple_annotations(
        args.mask_dir,
        args.image_dir,
        args.output_json,
        args.num_points,
        args.cluster_eps,
        args.match_thresh
    )

if __name__ == "__main__":
    main()