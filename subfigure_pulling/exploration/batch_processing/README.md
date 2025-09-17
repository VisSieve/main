# Introduction
This folder contains the code that was used to process pngs that were created by the pdffigures2 step and annotate them with bounding boxes and chart type guesses.

The code here isn't functional since it depends on a container and a number of other elements. But it does represent how things were accomplished in an adhoc fasion.

# Steps
* a user would python execute the `start_sbatch_processing.py` which would submit 50 individual sbatch requests with the index of that job passed in as an argument to identify the `worker` that's processing individual pngs. The bash script these sbatch jobs run is `coordinator.sh`
* `coordinator.sh` is responsible for using the CLI invocation that the visimages/visimages-detection  github repo shows and it passes along the worker index value to the `demo/image_demo.py` script
* `image_demo.py` is responsible for working on a set of about 200 images at a time, and it uses a pre-determined absolute path to the location where the png images are after the pdffigures2 step completes.
  * this script has lines of code that were modified in order to make it work in a batch workflow, and in the future the goal is to avoid using this at all in favor of the built in python interpreter methods for visimage detection of bounding boxes in images
```
from mmdet.apis import init_detector, inference_detector

config_file = 'work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/faster_rcnn_r50_fpn_1x_Visualization_Detect.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/visimages-best.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```
