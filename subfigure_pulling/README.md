# Figure analysis and subsetting

This step involves working with anything that was pulled from pdfs using pdffigures 2. Most of the work here took place in adhoc notebooks that will be maintained in the exploration folder.

The goal is to distill from these actual mature processing tools that accomplish the same results in a more reproducible fashion.


## Testing the code for the step

First step is to grab the visimages_mmdet container from here 

```
singularity build visimages_mmdet.sif docker://ghcr.io/devinbayly/visimages_mmdet:latest
```

Then provide an images path when running this exec line, and a threshold of probability for bounding box to be a suitable chart

```
singularity exec visimages_mmdet.sif python3 test_script.py <figure_path> <probability>
```
