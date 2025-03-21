

## Project Structure

```python
├─lib
│ ├─config/default   # configuration of training and validation
│ ├─core    
│ │ ├─activations.py   # activation function
│ │ ├─evaluate.py   # calculation of metric
│ │ ├─function.py   # training and validation of model
│ │ ├─general.py   #calculation of metric、nms、conversion of data-format、visualization
│ │ ├─loss.py   # loss function
│ │ ├─postprocess.py   # postprocess(refine da-seg and ll-seg, unrelated to paper)
│ ├─dataset
│ │ ├─AutoDriveDataset.py   # Superclass dataset，general function
│ │ ├─bdd.py   # Subclass dataset，specific function
│ │ ├─convect.py 
│ │ ├─DemoDataset.py   # demo dataset(image, video and stream)
│ ├─models
│ │ ├─YOLOP.py    # Setup and Configuration of model
│ │ ├─YOLOX_Head.py    # YOLOX's decoupled Head
│ │ ├─YOLOX_Loss.py    # YOLOX's detection Loss
│ │ ├─clr_head.py    # CLRNet
│ │ ├─clr_loss.py    # CLRNet loss
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
│ │ ├─dynamic_assign.py
│ │ ├─lane.py
│ │ ├─roi_gather.py
│ │ ├─visualization.py
│ │ ├─tusimple_metric.py
│ │ ├─split_dataset.py  # (Campus scene, unrelated to paper)
│ │ ├─plot.py  # plot_box_and_mask
│ │ ├─utils.py  # logging、device_select、time_measure、optimizer_select、model_save&initialize 、Distributed training
│ ├─run
│ │ ├─dataset/training time  # Visualization, logging and model_save
├─tools
│ │ ├─demo.py    # demo(folder、camera)
│ │ ├─test.py    
│ │ ├─train.py    
├─weights    # Pretraining model
```

---

## Requirement

This codebase has been developed with python version 3.7, PyTorch 1.12+ and torchvision 0.13+
```setup
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```setup
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
See `requirements.txt` for additional dependencies and version requirements.
```setup
pip install -r requirements.txt
```

## Dataset
For WoodScape: [Google Drive](https://drive.google.com/drive/folders/1ltj1QSNQJhThv8DVemM_l-G-GIH3JjMb)
### Transform txt2json
```bash
python tools/txt2json.py -i /path/to/original_dataset/box_2d_annotations -o /path/to/original_dataset/box_2d_json_annotations
```
### Splitting data
```bash
python tools/data_split.py --input_rgbLabels /path/to/original_dataet/semantic_annotations/rgbLabels \
                 --input_rgbImages /path/to/original_dataset/rgb_images \
                 --input_2DBox /path/to/original_dataset/box_2d_json_annotations \
                 --output_base_dir segmented_classes \
                 --output customized_dataset \
                 --train_ratio 0.9 \
                 --num_workers 8 \
                 --copy_files
```
### Generate curb points
```bash
# for train
python tools/gen_regression_annotations.py --mask_dir /path/to/customized_dataset/curb_seg_annotations/train/ --image_dir /path/to/customized_dataset/rgb_images/train/ --output_json /path/to/customized_dataset/rgb_images/train.json
# for val
python tools/gen_regression_annotations.py --mask_dir /path/to/customized_dataset/curb_seg_annotations/val/ --image_dir /path/to/customized_dataset/rgb_images/val/ --output_json /path/to/customized_dataset/rgb_images/val.json
```
We recommend the dataset directory structure to be the following:

## Dataset Structure

```python
# The id represent the correspondence relation
├─original_dataset # Download from source
| ├─rgb_images
| ├─box_2d_annotations
| ├─box_2d_json_annotations #from txt2json.py
| ├─semantic_annotations
| │ ├─rgbLabels
| | ├─gtLabels
├─customized_dataset #from data_split.py
│ ├─rgb_images
│ │ ├─train
│ │ ├─val
│ │ ├─train.json
│ │ ├─val.json
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─curb_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─person_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─vehicle_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─curb_reg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./lib/config/default.py`.

## Training

```shell
python tools/train.py
```

## Evaluation

```shell
python tools/test.py --weights weights/epoch-195.pth
```

## Demo

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --weights weights/epoch-195.pth
                     --source inference/image
                     --save-dir inference/image_output
                     --conf-thres 0.3
                     --iou-thres 0.45
```


## Acknowledgements

Our work would not be complete without the wonderful work of the following authors:

* [YOLOP](https://github.com/hustvl/YOLOP)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [HybridNets](https://github.com/datvuthanh/HybridNets)
* [YOLOPX](https://github.com/datvuthanh/HybridNets)
* [CLRNet](https://github.com/datvuthanh/HybridNets)


#step, split test and images 
txt2json
data_split
lane_regression_gen
