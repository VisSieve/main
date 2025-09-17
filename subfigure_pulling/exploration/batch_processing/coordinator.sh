#!/bin/bash
worker=$1
singularity exec --overlay working_overlay_success mmdetection.sif python demo/image_demo.py demo/demo.jpg work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/faster_rcnn_r50_fpn_1x_Visualization_Detect.py work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/visimages-best.pth --device cpu --worker $worker

