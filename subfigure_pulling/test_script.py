from mmdet.apis import init_detector, inference_detector

config_file = '/visimages-detection/work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/faster_rcnn_r50_fpn_1x_Visualization_Detect.py'
checkpoint_file = '/visimages-detection/work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/visimages-best.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
res = inference_detector(model, '/visimages-detection/demo/demo.jpg')
print(res)
