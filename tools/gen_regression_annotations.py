import os
import cv2
import json
import numpy as np
from sklearn.cluster import DBSCAN

def extract_lanes_from_mask(mask, num_points=72, cluster_eps=20, match_thresh=60):

    H, W = mask.shape
    h_samples = np.linspace(0.2 * H, 0.98*H, num_points, dtype=int).tolist()
    lanes = [[] for _ in range(num_points)]
    active_lanes = []  

    for i, y in enumerate(h_samples):
        x_positions = np.where(mask[y, :] > 0)[0]
        if len(x_positions) > 0:
            x_positions = x_positions.reshape(-1, 1)
            clustering = DBSCAN(eps=cluster_eps, min_samples=2).fit(x_positions)
            lane_clusters = []

            for label in np.unique(clustering.labels_):
                if label == -1:  
                    continue
                cluster = x_positions[clustering.labels_ == label].flatten()
                avg_x = int(np.mean(cluster))
                lane_clusters.append(avg_x)

            lane_clusters.sort()

            if not active_lanes:  
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
                        new_active_lanes.append([x])  

                active_lanes = new_active_lanes

            lanes[i] = [lane[-1] if lane else -2 for lane in active_lanes]  
        else:
            lanes[i] = [-2] * len(active_lanes) 

   
    num_lanes = max(len(lane) for lane in lanes)
    formatted_lanes = [[-2] * num_points for _ in range(num_lanes)]

    for i, lane_xs in enumerate(lanes):
        for j, x in enumerate(lane_xs): 
            if j < num_lanes:
                formatted_lanes[j][i]= x  # map detected x-coordinates into their lane
    
    # if no lanes were detected, return only -2s
    if num_lanes == 0:
        formatted_lanes = [[-2] * num_points]
    # print(np.asarray(formatted_lanes).shape)
    # print(len(formatted_lanes))
    return formatted_lanes, h_samples

def generate_tusimple_annotations(mask_dir, image_dir, output_json):

    annotations = []
    max = -1
    for mask_filename in sorted(os.listdir(mask_dir)):
        if not mask_filename.endswith(('.png', '.jpg')):  
            continue

        mask_path = os.path.join(mask_dir, mask_filename)
        image_path = os.path.join(image_dir, mask_filename)  
        # if image_path != '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/val/00030_RV.png':
        #     continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
        
        lanes, h_samples = extract_lanes_from_mask(mask)
        if len(lanes) > max:
            max = len(lanes)
            print(max, image_path)
        annotation = {
            "lanes": lanes,
            "h_samples": h_samples,
            "raw_file": image_path  
        }
        annotations.append(annotation)

    with open(output_json, 'w') as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + '\n')

mask_folder = "/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/ll_seg_annotations/train"
image_folder = "/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/train"
output_json_file = "/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/train.json"

generate_tusimple_annotations(mask_folder, image_folder, output_json_file)
