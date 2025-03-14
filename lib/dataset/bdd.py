import numpy as np
import json
import os.path as osp
from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm
from lib.utils.tusimple_metric import LaneEval
import random
import os
single_cls = True       # just detect vehicle
SPLIT_FILES = {
    'train+val': ['train.json', 'val.json'],
    'train': ['train.json'],
    'val': ['val.json'],
}


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, split, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg
        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.data_root = "/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/"
   
        self.h_samples = np.linspace(0.2 * 966, 0.98*966, 72, dtype=int).tolist()
        self.ori_img_h = 966
        self.ori_img_w = 1280

    def load_annotations(self):
        self.data_infos = []
        max_lanes= 0
        for anno_file in self.anno_files:
            anno_file = osp.join("/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/", anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('images', 'new_lane')[:-3] + '.png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x>=0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane)>0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append(
                    {
                        'img_path':
                        osp.join("/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/", data['raw_file']),
                        'img_name':
                        data['raw_file'],
                        'mask_path':
                        osp.join("/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/new_lane/", mask_path),
                        'lanes':
                        lanes,
                    }
                )
        if self.is_train:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes
    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        # for mask in tqdm(list(self.mask_list)[0:200] if self.is_train==True else list(self.mask_list)[0:30]):
        # for mask in tqdm(list(self.mask_list)[0:20000] if self.is_train==True else list(self.mask_list)[0:3000]):
        
        
        #################
        lane_reg_path = self.lane_reg_root
        with open(lane_reg_path, 'r') as anno_obj:
            lines = anno_obj.readlines()
        max_lanes = 0
        lane_reg_annotations = {}
        for line in lines:
            data = json.loads(line)
            lane_reg_annotations[data['raw_file']] = data
        ####################

        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root))
            person_path = mask_path.replace(str(self.mask_root), str(self.person_root))
            vehicle_path = mask_path.replace(str(self.mask_root), str(self.vehicle_root))
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))

            #################### for line in lines:
            data = lane_reg_annotations[image_path]
            # data = json.loads(line)
            # print(data)
            y_samples = data['h_samples']
            gt_lanes = data['lanes']
            # mask_path = data['raw_file'].replace('images', 'll_seg_annotations')[:-3] + '.png'
            lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x>=0] for lane in gt_lanes]

            lanes = [lane for lane in lanes if len(lane)>0]
            max_lanes = max(max_lanes, len(lanes))
            ####################


            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                    if single_cls:
                         cls_id=0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'person': person_path,
                'vehicle': vehicle_path,
                'lane': lane_path,
                'lane_reg': lanes
            }]
            self.max_lanes = max_lanes
            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain
    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.ori_img_h
        lanes = []
        # print(pred)
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            # print(xs)
            lane = (xs * self.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes
    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions,
                                                        runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate_lane(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir,
                                     'lane_reg_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename,
                                                self.cfg.test_json_file)
        return acc, result

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
