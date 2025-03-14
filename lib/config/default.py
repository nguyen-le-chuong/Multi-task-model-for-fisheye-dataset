import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR =  'runs/'
_C.GPUS = (0,)     # 显卡数 = len(GPUS)
_C.WORKERS = 8      # 指数据装载时cpu所使用的线程数，默认为8（注意，一般默使用8的话，会报错~~。原因是爆系统内存）
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.AUTO_RESUME =False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 4

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']

# MODIFY
_C.MODEL.PRETRAINED = ''

#

_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 2.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.PERSON_SEG_GAIN = 0.2
_C.LOSS.PERSON_IOU_GAIN = 0.2
_C.LOSS.VEHICLE_SEG_GAIN = 0.2

_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images'       # the path of images folder
_C.DATASET.LABELROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/det_annotations'      # the path of det_annotations folder
_C.DATASET.MASKROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/da_seg_annotations'                # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/ll_seg_annotations'               # the path of ll_seg_annotations folder
_C.DATASET.PERSONROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/person_seg_annotations'           # the path of person_seg_annotations folder
_C.DATASET.VEHICLEROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/vehicle_seg_annotations'         # the path of vehicle_seg_annotations folder
_C.DATASET.LANEREGROOT = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images'
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'png'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [966, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add

# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.1  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 80

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

_C.mosaic_rate = 0.0  # 1.0
_C.mixup_rate = 0.15

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task

_C.TRAIN.PLOT = False                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 16
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD  = 0.001
_C.TEST.NMS_IOU_THRESHOLD  = 0.6


# lane regression
_C.iou_loss_weight = 2.
_C.cls_loss_weight = 6.
_C.xyt_loss_weight = 0.5
_C.seg_loss_weight = 1.0
_C.img_w = 640
_C.img_h = 512
_C.num_classes = 13 + 1
_C.cut_height = 0
_C.num_points = 72
_C.max_lanes = 13
_C.test_parameters = CN(new_allowed=True)
_C.test_parameters.conf_threshold=0.40
_C.test_parameters.nms_thres=50
_C.test_parameters.nms_topk=13
_C.test_json_file = '/mnt/mmlab2024nas/huycq/chuong/temp/YOLOP/data/woodscape_m/images/val.json'


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
