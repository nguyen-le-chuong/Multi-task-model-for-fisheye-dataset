

## Project Structure

```python
├─inference
│ ├─image   # inference images
│ ├─image_output   # inference result
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
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
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

## Pre-trained Model
You can get the pre-trained model from <a href="https://pan.baidu.com/s/1k7_M8vrgQCnlY-FlaA0g6Q">baidu</a> or <a href="https://drive.google.com/file/d/1dlwaElu0dQQdoEeJkuP2LKGx1TSCjE-z/view?usp=sharing">google</a>.
Baidu extraction code：fvuc

## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
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

## License

YOLOPX is released under the [MIT Licence](LICENSE).

## Acknowledgements

Our work would not be complete without the wonderful work of the following authors:

* [YOLOP](https://github.com/hustvl/YOLOP)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [HybridNets](https://github.com/datvuthanh/HybridNets)

