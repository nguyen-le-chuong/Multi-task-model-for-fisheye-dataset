import time
from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask,plot_one_box,show_seg_result
from lib.utils.visualization import imshow_lanes
import torch
from threading import Thread
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
import random
import cv2
import os
import math
from torch.cuda import amp
from tqdm import tqdm
import itertools
def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank = -1):
    """
    train for one epoch

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:
    outputs(2,)
    output[0] len:3, [1,3,32,32,85], [1,3,16,16,85], [1,3,8,8,85]
    output[1] len:1, [2,256,256]
    output[2] len:1, [2,256,256]
    target(2,)
    target[0] [1,n,5]
    target[1] [2,256,256]
    target[2] [2,256,256]
    Returns:
    None

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # print(device)
    # switch to train mode
    model.train()
    start = time.time()

    for i, (input, target, paths, shapes) in enumerate(train_loader):
        # print(i)
        # if i < 820:
        #     continue
        intermediate = time.time()
        #print('tims:{}'.format(intermediate-start))
        num_iter = i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            # warm up
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for k, tgt in enumerate(target):
                if k != len(target) - 1:
                    assign_target.append(tgt.to(device))
                else:
                    assign_target.append(tgt)
            target = assign_target
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            # print(outputs)
            total_loss, head_losses = criterion(outputs, target, shapes,model,input)            
        det_all_loss, da_seg_loss, person_seg_loss, person_tversky_loss, vehicle_seg_loss, ll_seg_loss, ll_tversky_loss, reg_loss, loss = head_losses
        # compute gradient and do update step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # print(rank)
        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.update(total_loss.item(), input.size(0))

            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            end = time.time()
            # print(i)
            if i % cfg.PRINT_FREQ == 0:
                msg = (
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Speed {speed:.1f} samples/s\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Total Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                    'Det Loss {det_loss:.5f}\t'
                    'DA Seg Loss {da_seg_loss:.5f}\t'
                    'Person Seg Loss {person_seg_loss:.5f}\t'
                    'Person Tversky Loss {person_tversky_loss:.5f}\t'
                    'Vehicle Seg Loss {vehicle_seg_loss:.5f}\t'
                    'Lane Seg Loss {ll_seg_loss:.5f}\t'
                    'Lane Tversky Loss {ll_tversky_loss:.5f}\t'
                    'Lane Regression Loss {ll_reg_loss:.5f}'
                ).format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    speed=input.size(0)/batch_time.val,
                    data_time=data_time,
                    loss=losses,
                    det_loss=det_all_loss,
                    da_seg_loss=da_seg_loss,
                    person_seg_loss=person_seg_loss,
                    person_tversky_loss=person_tversky_loss,
                    vehicle_seg_loss=vehicle_seg_loss,
                    ll_seg_loss=ll_seg_loss,
                    ll_tversky_loss=ll_tversky_loss, 
                    ll_reg_loss = reg_loss
                )

                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('det_loss', det_all_loss, global_steps)
                writer.add_scalar('da_seg_loss', da_seg_loss, global_steps)
                writer.add_scalar('person_seg_loss', person_seg_loss, global_steps)
                writer.add_scalar('person_tversky_loss', person_tversky_loss, global_steps)
                writer.add_scalar('vehicle_seg_loss', vehicle_seg_loss, global_steps)
                writer.add_scalar('lane_seg_loss', ll_seg_loss, global_steps)
                writer.add_scalar('lane_tversky_loss', ll_tversky_loss, global_steps)
                writer.add_scalar('lane_regression_loss', reg_loss, global_steps)

                writer_dict['train_global_steps'] = global_steps + 1
            # if i % cfg.PRINT_FREQ == 0:
            #     msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #           'Speed {speed:.1f} samples/s\t' \
            #           'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #           'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
            #               epoch, i, len(train_loader), batch_time=batch_time,
            #               speed=input.size(0)/batch_time.val,
            #               data_time=data_time, loss=losses)
                
            #     logger.info(msg)

            #     writer = writer_dict['writer']
            #     global_steps = writer_dict['train_global_steps']
            #     writer.add_scalar('train_loss', losses.val, global_steps)
            #     # writer.add_scalar('train_acc', acc.val, global_steps)
            #     writer_dict['train_global_steps'] = global_steps + 1



def validate(epoch,config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return 
    - writer_dict: 

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None

    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # print(save_dir)
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE] #imgsz is multiple of max_stride
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False #is coco dataset
    save_conf=False # save auto-label confidences
    verbose=False
    save_hybrid=False
    log_imgs,wandb = min(16,100), None

    nc = 1
    iouv = torch.linspace(0.5,0.95,10).to(device)     #iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen =  0 
    confusion_matrix = ConfusionMatrix(nc=model.nc) #detector confusion matrix
    da_metric = SegmentationMetric(config.num_seg_class) #segment confusion matrix    
    person_metric = SegmentationMetric(2) #segment confusion matrix
    vehicle_metric = SegmentationMetric(config.num_seg_class) #segment confusion matrix
    ll_metric = SegmentationMetric(2) #segment confusion matrix 
    

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    
    losses = AverageMeter()

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    person_acc_seg = AverageMeter()
    person_IoU_seg = AverageMeter()
    person_mIoU_seg = AverageMeter()

    vehicle_acc_seg = AverageMeter()
    vehicle_IoU_seg = AverageMeter()
    vehicle_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    # switch to eval mode
    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    lane_predictions = []
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for k, tgt in enumerate(target):
                if k != len(target) - 1:
                    assign_target.append(tgt.to(device))
                else:
                    assign_target.append(tgt)
            target = assign_target
            nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]

            t = time_synchronized()
            det_out, da_seg_out, person_seg_out, vehicle_seg_out, ll_seg_out, (lane_reg_out, lan_reg_lists) = model(img)
            lane_predictions.extend(lane_reg_out)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0),img.size(0))

            inf_out,train_out = det_out
            
            #driving area segment evaluation
            _,da_predict=torch.max(da_seg_out, 1)
            _,da_gt=torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,img.size(0))
            da_IoU_seg.update(da_IoU,img.size(0))
            da_mIoU_seg.update(da_mIoU,img.size(0))

            #person segment evaluation
            _,person_predict=torch.max(person_seg_out, 1)
            _,person_gt=torch.max(target[2], 1)
            person_predict = person_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            person_gt = person_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            person_metric.reset()
            person_metric.addBatch(person_predict.cpu(), person_gt.cpu())
            person_acc = person_metric.pixelAccuracy()
            person_IoU = person_metric.IntersectionOverUnion()
            person_mIoU = person_metric.meanIntersectionOverUnion()

            person_acc_seg.update(person_acc,img.size(0))
            person_IoU_seg.update(person_IoU,img.size(0))
            person_mIoU_seg.update(person_mIoU,img.size(0))

            #vehicle segment evaluation
            _,vehicle_predict=torch.max(vehicle_seg_out, 1)
            _,vehicle_gt=torch.max(target[3], 1)
            vehicle_predict = vehicle_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            vehicle_gt = vehicle_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            vehicle_metric.reset()
            vehicle_metric.addBatch(vehicle_predict.cpu(), vehicle_gt.cpu())
            vehicle_acc = vehicle_metric.pixelAccuracy()
            vehicle_IoU = vehicle_metric.IntersectionOverUnion()
            vehicle_mIoU = vehicle_metric.meanIntersectionOverUnion()

            vehicle_acc_seg.update(vehicle_acc,img.size(0))
            vehicle_IoU_seg.update(vehicle_IoU,img.size(0))
            vehicle_mIoU_seg.update(vehicle_mIoU,img.size(0))


            #lane line segment evaluation
            _,ll_predict=torch.max(ll_seg_out, 1)
            _,ll_gt=torch.max(target[4], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc,img.size(0))
            ll_IoU_seg.update(ll_IoU,img.size(0))
            ll_mIoU_seg.update(ll_mIoU,img.size(0))
            #lane 
            
            total_loss, head_losses = criterion((train_out,da_seg_out, person_seg_out, vehicle_seg_out, ll_seg_out, lan_reg_lists), target, shapes,model, img )   #Compute loss
            losses.update(total_loss.item(), img.size(0))

            #NMS         
            t = time_synchronized()
            # target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = []  # for autolabelling
            output = non_max_suppression(inf_out, conf_thres= config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
            #output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
            #output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRES, iou_thres=config.TEST.NMS_IOU_THRES)
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))
            if config.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(test_batch_size):
                        img_test = cv2.imread(paths[i])
                        da_seg_mask = da_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_seg_mask = torch.max(da_seg_mask, 1)

                        da_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_gt_mask = torch.max(da_gt_mask, 1)

                        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_test1 = img_test.copy()
                        _ = show_seg_result(img_test, da_seg_mask, i,epoch,save_dir, seg_type="da")
                        _ = show_seg_result(img_test1, da_gt_mask, i, epoch, save_dir, seg_type="da", is_gt=True)

                        img_person = cv2.imread(paths[i])
                        person_seg_mask = person_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        person_seg_mask = torch.nn.functional.interpolate(person_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, person_seg_mask = torch.max(person_seg_mask, 1)

                        person_gt_mask = target[2][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        person_gt_mask = torch.nn.functional.interpolate(person_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, person_gt_mask = torch.max(person_gt_mask, 1)

                        person_seg_mask = person_seg_mask.int().squeeze().cpu().numpy()
                        person_gt_mask = person_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_person1 = img_person.copy()
                        _ = show_seg_result(img_person, person_seg_mask, i,epoch,save_dir, seg_type="person")
                        _ = show_seg_result(img_person1, person_gt_mask, i, epoch, save_dir, is_gt=True, seg_type="person")

                        img_vehicle = cv2.imread(paths[i])
                        vehicle_seg_mask = vehicle_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        vehicle_seg_mask = torch.nn.functional.interpolate(vehicle_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, vehicle_seg_mask = torch.max(vehicle_seg_mask, 1)

                        vehicle_gt_mask = target[3][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        vehicle_gt_mask = torch.nn.functional.interpolate(vehicle_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, vehicle_gt_mask = torch.max(vehicle_gt_mask, 1)

                        vehicle_seg_mask = vehicle_seg_mask.int().squeeze().cpu().numpy()
                        vehicle_gt_mask = vehicle_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_vehicle1 = img_vehicle.copy()
                        _ = show_seg_result(img_vehicle, vehicle_seg_mask, i,epoch,save_dir, seg_type="vehicle")
                        _ = show_seg_result(img_vehicle1, vehicle_gt_mask, i, epoch, save_dir, is_gt=True, seg_type="vehicle")


                        img_ll = cv2.imread(paths[i])
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)

                        ll_gt_mask = target[4][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i,epoch,save_dir, seg_type="ll")
                        _ = show_seg_result(img_ll1, ll_gt_mask, i, epoch, save_dir, seg_type="ll", is_gt=True)

                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = output[i].clone()
                        if len(det):
                            det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_det.shape).round()
                        for *xyxy,conf,cls in reversed(det):
                            #print(cls)
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir+"/batch_{}_det_pred.png".format(i),img_det)

                        lane_reg_pred = lane_reg_out[i]
                        # for lane_pred in lane_reg_pred:
                        lanes = [lane.to_array(config) for lane in lane_reg_pred]
                        imshow_lanes(img_gt, lanes, out_file=save_dir+"/batch_{}_lane_reg_pred.png".format(i))

                        # labels = target[0][target[0][:, 0] == i, 1:]
                        # # print(labels)
                        # labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                        # if len(labels):
                        #     labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_gt.shape).round()
                        # for cls,x1,y1,x2,y2 in labels:
                        #     #print(names)
                        #     #print(cls)
                        #     label_det_gt = f'{names[int(cls)]}'
                        #     xyxy = (x1,y1,x2,y2)
                        #     plot_one_box(xyxy, img_gt , label=label_det_gt, color=colors[int(cls)], line_thickness=3)
                        # cv2.imwrite(save_dir+"/batch_{}_det_gt.png".format(i),img_gt)
        # Statistics per image
        # output([xyxy,conf,cls])
        # target[0] ([img_id,cls,xyxy])
        nlabel = (target[0].sum(dim=2) > 0).sum(dim=1)  # number of objects # [batch, num_gt ]
        for si, pred in enumerate(output):
            # gt per image
            nl = int(nlabel[si])
            labels = target[0][si, :nl, 0:5] # [ num_gt_per_image, [cx, cy, w, h] ]
            # gt_classes = target[0][si, :nl, 0]   

            # labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image 
            # nl = num_gt
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if config.TEST.SAVE_JSON:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if config.TEST.PLOTS:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if config.TEST.PLOTS and batch_i < 3:
            f = save_dir +'/'+ f'test_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
            f = save_dir +'/'+ f'test_batch{batch_i}_pred.jpg'  # predictions
            #Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
        ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    #print(map70)
    #print(map75)

    # Print results per class
    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    person_segment_result = (person_acc_seg.avg,person_IoU_seg.avg,person_mIoU_seg.avg)
    vehicle_segment_result = (vehicle_acc_seg.avg,vehicle_IoU_seg.avg,vehicle_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)

    # print(da_segment_result)
    # print(ll_segment_result)
    detect_result = np.asarray([mp, mr, map50, map])
    # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
    #print segmet_result
    t = [T_inf.avg, T_nms.avg]

    ###################lane regression
    results_lane, metric_lane = val_loader.dataset.evaluate_lane(lane_predictions, save_dir)

    return da_segment_result, person_segment_result, vehicle_segment_result, ll_segment_result, detect_result, losses.avg, maps, t, metric_lane, results_lane
        


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0