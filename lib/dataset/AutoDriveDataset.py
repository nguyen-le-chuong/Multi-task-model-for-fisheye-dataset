import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
import albumentations as A
from collections import OrderedDict
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from torch.nn.utils.rnn import pad_sequence
import math
import os
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def visualize_and_save(image, seg_label, person_label, vehicle_label, lane_label, labels, lane_reg_label, output_folder, filename):
    """
    Visualizes and saves the augmented image with overlays of segmentation masks, bounding boxes, and lane regression keypoints.

    Args:
        image (numpy array): Transformed image.
        seg_label (numpy array): Segmentation mask.
        person_label (numpy array): Person mask.
        vehicle_label (numpy array): Vehicle mask.
        lane_label (numpy array): Lane mask.
        labels (numpy array): Bounding boxes and class labels (format: [class_id, x_min, y_min, x_max, y_max]).
        lane_reg_label (list of list of tuples): Nested list of keypoints for lane regression.
        output_folder (str): Path to save the visualized image.
        filename (str): Filename for the saved image.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert masks to color overlays
    seg_overlay = cv2.applyColorMap((seg_label * 255).astype(np.uint8), cv2.COLORMAP_JET)
    person_overlay = cv2.applyColorMap((person_label * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    vehicle_overlay = cv2.applyColorMap((vehicle_label * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
    lane_overlay = cv2.applyColorMap((lane_label * 255).astype(np.uint8), cv2.COLORMAP_SPRING)

    # Blend the masks with the image
    blended = cv2.addWeighted(image, 0.7, seg_overlay, 0.3, 0)
    blended = cv2.addWeighted(blended, 0.7, person_overlay, 0.3, 0)
    blended = cv2.addWeighted(blended, 0.7, vehicle_overlay, 0.3, 0)
    blended = cv2.addWeighted(blended, 0.7, lane_overlay, 0.3, 0)

    # Draw bounding boxes on the image
    for label in labels:
        class_id, x_min, y_min, x_max, y_max = label
        color = (0, 255, 0)  # Green for bounding boxes
        cv2.rectangle(blended, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(blended, str(int(class_id)), (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw keypoints for lane regression
    lane_colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255)]  # Distinct colors for different lanes
    for lane_idx, lane in enumerate(lane_reg_label):
        color = lane_colors[lane_idx % len(lane_colors)]  # Cycle through colors for lanes
        for point in lane:
            x, y = point
            cv2.circle(blended, (int(x), int(y)), 3, color, -1)  # Draw keypoint
            cv2.putText(blended, f"{lane_idx}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save the visualization
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, blended)
    print(f"Saved visualization to {save_path}")
class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        person_root = Path(cfg.DATASET.PERSONROOT)
        vehicle_root = Path(cfg.DATASET.VEHICLEROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        lane_reg_root = Path(cfg.DATASET.LANEREGROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.person_root = person_root / indicator
        self.vehicle_root = vehicle_root / indicator
        self.lane_root = lane_root / indicator
        print(indicator)
        self.lane_reg_root = lane_reg_root / f"{indicator}.json"
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        # albumentation data arguments
        self.albumentations_transform = A.Compose([

            A.OneOf([
                A.MotionBlur(p=0.1),
                A.MedianBlur(p=0.1),
                A.Blur(p=0.1),
            ], p=0.2),

            A.GaussNoise(p=0.02),
            A.CLAHE(p=0.02),
            A.RandomBrightnessContrast(p=0.02),
            A.RandomGamma(p=0.02),
            A.ImageCompression(quality_lower=75, p=0.02),

            A.OneOf([
                A.RandomSnow(p=0.1),  # 加雪花
                A.RandomRain(p=0.1),  # 加雨滴
                A.RandomFog(p=0.1),  # 加雾
                A.RandomSunFlare(p=0.1),  # 加阳光
                A.RandomShadow(p=0.1),  # 加阴影
            ], p=0.2),

            A.OneOf([
                A.ToGray(p=0.1),
                A.ToSepia(p=0.1),
            ], p=0.2),

            ],
            
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            additional_targets={'mask': 'mask', 'mask0': 'mask0', 'mask1': 'mask1', 'mask2': 'mask2'})

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.mosaic_border = [-192, -320]

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

        self.mosaic_rate = cfg.mosaic_rate
        self.mixup_rate = cfg.mixup_rate

        self.refine_layers = 3
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]
        # print('x', x)
        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))
        # all_xs = np.hstack(interp_xs)
        # print('after', all_xs)
        # print(len(all_xs))
        # print('after', len(all_xs_int))
        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image
    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane
    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes
    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines
    def transform_annotation(self, old, img_wh=None):
        img_w, img_h = self.img_w, self.img_h
        # print(img_w, img_h)
        # old_lanes = old
        old_lanes = old
        # print(old)
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        # print(old_lanes)
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        lanes_endpoints = np.ones((self.max_lanes, 2))
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            # print(len(xs_outside_image))
            all_xs = np.hstack((xs_outside_image, xs_inside_image))	
            # all_xs = np.hstack(xs_inside_image)	
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)
            
            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        new_lanes = lanes
        return new_lanes    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def load_mosaic(self, idx):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        lane_reg_labels4 = []
        w_mosaic, h_mosaic = 640, 512

        yc = int(random.uniform(-self.mosaic_border[0], 2 * h_mosaic + self.mosaic_border[0])) # 192,3x192
        xc = int(random.uniform(-self.mosaic_border[1], 2 * w_mosaic + self.mosaic_border[1])) # 320,3x320
        
        indices = range(len(self.db))
        indices = [idx] + random.choices(indices, k=3)  # 3 additional iWmage indices
                        
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            # img, labels, seg_label, (h0,w0), (h, w), path = self.load_image(index), h=384, w = 640
            img, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_labels, (h0, w0), (h,w), path  = self.load_image(index)
                        
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((h_mosaic * 2, w_mosaic * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

                seg4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles
                
                person4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles

                vehicle4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles

                lane4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles
                # 大图中左上角、右下角的坐标
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 小图中左上角、右下角的坐标
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_mosaic * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_mosaic * 2), min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            seg4[y1a:y2a, x1a:x2a] = seg_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            person4[y1a:y2a, x1a:x2a] = person_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            vehicle4[y1a:y2a, x1a:x2a] = vehicle_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            lane4[y1a:y2a, x1a:x2a] = lane_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b
            
            if len(labels):
                labels[:, 1] += padw
                labels[:, 2] += padh
                labels[:, 3] += padw
                labels[:, 4] += padh
            
                labels4.append(labels)
            if len(lane_reg_labels):
                lane_reg_labels = [[(x + padw, y + padh) for (x, y) in lane] for lane in lane_reg_labels]
                lane_reg_labels4.append(lane_reg_labels)
        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        
        new = labels4.copy()
        new[:, 1:] = np.clip(new[:, 1:], 0, 2*w_mosaic)
        new[:, 2:5:2] = np.clip(new[:, 2:5:2], 0, 2*h_mosaic)

        # filter candidates
        i = box_candidates(box1=labels4[:,1:5].T, box2=new[:,1:5].T)
        labels4 = labels4[i]
        labels4[:] = new[i] 
        if lane_reg_labels4:
            new_lanes = []
            for lane in lane_reg_labels4:
                new_lane = [(np.clip(x, 0, 2 * w_mosaic), np.clip(y, 0, 2 * h_mosaic)) for x, y in lane]
                new_lanes.append(new_lane)

            lane_reg_labels4 = new_lanes  # Update with clipped lane labels

        return img4, labels4, seg4, person4, vehicle4, lane4, lane_reg_labels4, (h0, w0), (h, w), path

    def mixup(self, im, labels, seg_label, person_label, vehicle_label, lane_label,lane_reg_label, im2, labels2, seg_label2, person_label2, vehicle_label2, lane_label2, lane_reg_label2):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        seg_label |= seg_label2
        person_label |= person_label2
        vehicle_label |= vehicle_label2
        lane_label |= lane_label2
        lane_reg_label.extend(lane_reg_label2)
        return im, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_label

    def load_image(self, idx):

        # yolopx
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_label = cv2.imread(data["mask"], 0)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        person_label = cv2.imread(data["person"], 0)
        vehicle_label = cv2.imread(data["vehicle"], 0)
        lane_label = cv2.imread(data["lane"], 0)
        # print(self.inputsize)
        
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            person_label = cv2.resize(person_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            vehicle_label = cv2.resize(vehicle_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
       
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()

            labels[:, 1] = (det_label[:, 1] - det_label[:, 3] / 2) * w
            labels[:, 2] = (det_label[:, 2] - det_label[:, 4] / 2) * h 
            labels[:, 3] = (det_label[:, 1] + det_label[:, 3] / 2) * w
            labels[:, 4] = (det_label[:, 2] + det_label[:, 4] / 2) * h
        
        lane_reg_label = data["lane_reg"]
        if lane_reg_label:
            # Resize lane points
            resized_lane_reg_label = []
            for lane in lane_reg_label:
                # Resize each point in the lane
                resized_lane = [(int(x * w / w0), int(y * h / h0)) for (x, y) in lane]
                resized_lane_reg_label.append(resized_lane)
            
            lane_reg_label = resized_lane_reg_label
        
        return img, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_label, (h0, w0), (h,w), data['image']

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """ 
        if self.is_train:
            mosaic_this = False
            if random.random() < self.mosaic_rate:
                mosaic_this = True
                #  this doubles training time with inherent stuttering in tqdm, prob cpu or io bottleneck, does prefetch_generator work with ddp? (no improvement)
                #  updated, mosaic is inherently slow, maybe cache the images in RAM? maybe it was IO bottleneck of reading 4 images everytime? time it
                img, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_label, (h0, w0), (h, w), path = self.load_mosaic(idx)

                # mixup is double mosaic, really slow
                if random.random() < self.mixup_rate:
                    img2, labels2, seg_label2, person_label2, vehicle_label2, lane_label2, lane_reg_label2, (_, _), (_, _), _ = self.load_mosaic(random.randint(0, len(self.db) - 1))
                    img, labels, seg_label, person_label, vehicle_label, lane_label = self.mixup(img, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_label, img2, labels2, seg_label2, person_label2, vehicle_label2, lane_label2, lane_reg_label2)
            else:

                img, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_label, (h0, w0), (h,w), path  = self.load_image(idx)
            # # print(person_label.shape)
            # print(lane_label.shape)
            # line_strings_org = self.lane_to_linestrings(lane_reg_label)
            # line_strings_org = LineStringsOnImage(line_strings_org,
            #                                   shape=img.shape)
            # print(line_strings_org)
            try:
                # if lane_reg_label:
                #     all_keypoints = []
                #     lane_lengths = []
                #     for lane in lane_reg_label:
                #         lane_lengths.append(len(lane))
                #         all_keypoints.extend(lane)  # flatten all points
                new = self.albumentations_transform(image=img, mask=seg_label, mask0=person_label, mask1 = vehicle_label, mask2 = lane_label,
                                                    bboxes=labels[:, 1:] if len(labels) else labels,
                                                    class_labels=labels[:, 0] if len(labels) else labels)
                                                    # lanes_reg=all_keypoints if lane_reg_label else None)
                img = new['image']
                labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) if len(labels) else labels
                seg_label = new['mask']
                person_label = new['mask0']
                vehicle_label = new['mask1']
                lane_label = new['mask2']
                # if lane_reg_label:
                #     transformed_keypoints = new['lanes_reg']
                #     lanes_final_transformed = []
                #     idx = 0
                #     for length in lane_lengths:
                #         lane_pts = transformed_keypoints[idx: idx + length]
                #         lanes_final_transformed.append(lane_pts)
                #         idx += length
                #     lane_reg_label = lanes_final_transformed
            except ValueError:  # bbox have width or height == 0
                pass

            combination = (img, seg_label, person_label, vehicle_label, lane_label)
            (img, seg_label, person_label, vehicle_label, lane_label), labels, lane_reg_label = random_perspective(
                combination=combination,
                targets=labels,
                lane_reg=lane_reg_label,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR,
                border=self.mosaic_border if mosaic_this else (0, 0)
            )
            # output_folder = "./visualized_augmented_data"
            # filename = "augmented_image.jpg"
            # visualize_and_save(
            #     image=img,
            #     seg_label=seg_label,
            #     person_label=person_label,
            #     vehicle_label=vehicle_label,
            #     lane_label=lane_label,
            #     labels=labels,
            #     lane_reg_label=lane_reg_label if lane_reg_label else [],
            #     output_folder=output_folder,
            #     filename=filename
            # )

            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)

            # random left-right flip
            if random.random() < 0.5:
                img = np.fliplr(img)

                if len(labels):
                    rows, cols, channels = img.shape
                    x1 = labels[:, 1].copy()
                    x2 = labels[:, 3].copy()
                    x_tmp = x1.copy()
                    labels[:, 1] = cols - x2
                    labels[:, 3] = cols - x_tmp
                
                seg_label = np.fliplr(seg_label)
                person_label = np.fliplr(person_label)
                vehicle_label = np.fliplr(vehicle_label)
                lane_label = np.fliplr(lane_label)
                if lane_reg_label:
                    rows, cols, channels = img.shape
                    for i in range(len(lane_reg_label)):
                        lane_reg_label[i] = [(cols - x, y) for (x, y) in lane_reg_label[i]]
            # print(lane_reg_label)
            # random up-down flip
            # if random.random() < 0.0:
            #     img = np.flipud(img)

            #     if len(labels):
            #         rows, cols, channels = img.shape
            #         y1 = labels[:, 2].copy()
            #         y2 = labels[:, 4].copy()
            #         y_tmp = y1.copy()
            #         labels[:, 2] = rows - y2
            #         labels[:, 4] = rows - y_tmp

            #     seg_label = np.flipud(seg_label)
            #     person_label = np.flipud(person_label)
            #     vehicle_label = np.flipud(vehicle_label)
            #     lane_label = np.flipud(lane_label)
            #     if lane_reg_label:
            #         rows, cols, channels = img.shape
            #         for i in range(len(lane_reg_label)):
            #             lane_reg_label[i] = [(x, rows - y) for (x, y) in lane_reg_label[i]]
        
        else:
            img, labels, seg_label, person_label, vehicle_label, lane_label, lane_reg_label, (h0, w0), (h,w), path = self.load_image(idx)
        # print(img.shape)
        img_lane = img.copy()
        (img, seg_label, person_label, vehicle_label, lane_label), ratio, pad = letterbox((img, seg_label, person_label, vehicle_label, lane_label), 640, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # print('r', ratio)
        # print('p', pad)
        if len(labels):
            # update labels after letterbox
            labels[:, 1] = ratio[0] * labels[:, 1] + pad[0]
            labels[:, 2] = ratio[1] * labels[:, 2] + pad[1]
            labels[:, 3] = ratio[0] * labels[:, 3] + pad[0]
            labels[:, 4] = ratio[1] * labels[:, 4] + pad[1]     

            # convert xyxy to ( cx, cy, w, h )
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

        labels_out = torch.zeros((len(labels), 5))
        if len(labels):
            labels_out[:, :] = torch.from_numpy(labels)
        if len(lane_reg_label):
            lane_reg_label = [
                [(ratio[0] * x + pad[0], ratio[1] * y + pad[1]) for (x, y) in lane]
                for lane in lane_reg_label
            ]
        img = np.ascontiguousarray(img)

        if self.cfg.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        _,person1 = cv2.threshold(person_label, 1, 255, cv2.THRESH_BINARY)
        _,person2 = cv2.threshold(person_label, 1, 255, cv2.THRESH_BINARY_INV)
        _,vehicle1 = cv2.threshold(vehicle_label, 1, 255, cv2.THRESH_BINARY)
        _,vehicle2 = cv2.threshold(vehicle_label, 1, 255, cv2.THRESH_BINARY_INV)
        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)

        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)

        person1 = self.Tensor(person1)
        person2 = self.Tensor(person2)

        vehicle1 = self.Tensor(vehicle1)
        vehicle2 = self.Tensor(vehicle2)

        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
        person_label = torch.stack((person2[0], person1[0]), 0)
        vehicle_label = torch.stack((vehicle2[0], vehicle1[0]), 0)
        lane_label = torch.stack((lane2[0], lane1[0]),0)
        # print(lane_reg_label)
        line_strings_org = self.lane_to_linestrings(lane_reg_label)
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img.shape)
        line_strings_org.clip_out_of_image_()
        batch = self.linestrings_to_lanes(line_strings_org)
        # print(lane_reg_label)
        lane_reg_label = self.transform_annotation(batch)
#         output_file = 'labels.txt'

# # Write the label to the file
#         with open(output_file, 'a') as file:  # 'a' mode appends to the file if it already exists
#             file.write(f"{lane_reg_label}\n")
        target = [labels_out, seg_label, person_label, vehicle_label, lane_label, lane_reg_label]
        # img_lane = self.transform(img_lane)

        img = self.transform(img)

        # add lane regression
        # data_info = self.data_infos[idx]
        # img = cv2.imread(data_info['img_path'])
        # img = img[self.cfg.cut_height:, :, :]
        # sample = data_info.copy()
        return img, target, path, shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg, label_person, label_vehicle, label_lane, label_lane_reg = [], [], [], [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_person, l_vehicle, l_lane, l_lane_reg = l
            # l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_person.append(l_person)
            label_vehicle.append(l_vehicle)
            label_lane.append(l_lane)
            label_lane_reg.append(l_lane_reg)
        label_det = pad_sequence(label_det, batch_first = True, padding_value = 0)
        # label_lane_reg = pad_sequence(label_lane_reg, batch_first = True, padding_value = 0)
        return torch.stack(img, 0), [label_det, torch.stack(label_seg, 0), torch.stack(label_person, 0), torch.stack(label_vehicle, 0), torch.stack(label_lane, 0), label_lane_reg], paths, shapes


