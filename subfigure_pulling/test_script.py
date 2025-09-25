from mmdet.apis import init_detector, inference_detector
from PIL import Image
from pathlib import Path
# add single test image 
classes = ('flow_diagram', 'scatterplot', 'bar_chart', 'graph', 'treemap', 'table', 'line_chart', 'tree', 'small_multiple', 'heatmap', 'matrix', 'map', 'pie_chart', 'sankey_diagram', 'area_chart', 'proportional_area_chart', 'glyph_based', 'stripe_graph', 'parallel_coordinate', 'sunburst_icicle', 'unit_visualization', 'polar_plot', 'error_bar', 'box_plot', 'sector_chart', 'word_cloud', 'donut_chart', 'hierarchical_edge_bundling', 'chord_diagram', 'storyline')
figure = next(Path("/xdisk/chrisreidy/baylyd/vis_sieve/vis-sieve/pdf_grabbing/pdf_symlinks").glob("*Figure*.png"))

config_file = '/visimages-detection/work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/faster_rcnn_r50_fpn_1x_Visualization_Detect.py'

checkpoint_file = '/visimages-detection/work_dirs/faster_rcnn_r50_fpn_1x_Visualization_Detect/visimages-best.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# limit the export of subsections that aren't high enough prob
prob_threshold = .3

def subset_image(m,figure_path):
  # this returns a list of 30 elements (matching number of chart classes)
  # each element in the list is a probability for the match of that chart type and the bbox coords, so there's 5 elements in there (order is bbox, then prob)
  res = inference_detector(model, str(figure))
  print(res)
  im = Image.open(str(figure_path))
  # track the index of the loop so we can print out the name of hte chart type
  for class_ind,ctype in enumerate(res):
    if ctype.size == 0:
      continue
    # there might be multiple charts of this type so we will iterate over them
    # convert them to list instead of array
    # we will put the prob first and the bbox second so we can unpack when iterating
    # even though the prob is the last element in the list
    list_data =[[e[-1],list(e[:-1])] for e in ctype]
    for prob,bbox in list_data:
      if prob < prob_threshold:
        continue
      chart_type_name = classes[class_ind]
      print(chart_type_name,prob,bbox)
      part = im.crop(bbox)
      part.save(f"{figure_path.stem}_{chart_type_name}.png")



subset_image(model,figure)
      
