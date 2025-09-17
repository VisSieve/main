# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import inspect
import json
import tqdm
import random
from pathlib import Path
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector, init_detector, show_result_pyplot)
#from mmdet.apis import (async_inference_detector, inference_detector, init_detector)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('--worker',type=int,default =0, help='batch processing id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    figures = sorted(Path("/xdisk/chrisreidy/baylyd/vis_sieve/vis-sieve/pdf_grabbing/pdf_symlinks").glob("*Figure*.png"))
    out_folder = Path("outputs_temp")
    out_folder.mkdir(exist_ok=True)
    # test a single image
    #figures_subset = random.sample(figures,2000)
    figures_subset = figures[args.worker*200:(args.worker+1)*200]
    ## BREAKING CHANGE BELOW
    for img in tqdm.tqdm(figures_subset[:1]):
      model = init_detector(args.config, args.checkpoint, device=args.device)
      if hasattr(model,"module"):
        model  = model.module
      result = inference_detector(model, str(img))
      # set 0 arrays where the shape is 0,5
      result = [r if r.shape[0] >0 else np.zeros((1,5)) for r in result]
      print(img)
      print(result)
      print(len(result))
      [print("part",e.shape) for e in result]
      print(model.CLASSES)
      print(len(model.CLASSES))
      # padd the results
      bboxes = np.vstack(result)
      scores = bboxes[:,-1]
      print("these are the scores",scores)
      print([(s,c) for s,c in zip(scores,model.CLASSES)])
      # attempt to print the classes that go with each score
      #show_result_pyplot(
      #    model,
      #    str(img),
      #    result,
      #    palette="coco",
      #    score_thr = .3,
      #    out_file =f"{out_folder}/result_{img.stem}.jpg")
      ##print(result)
      ##print(result[0])
      #result_list = [a.tolist() for a in result]
      #Path(f"{out_folder}/array_{img.stem}.json").write_text(json.dumps(result_list))
     
    #result = inference_detector(model, args.img)
    ## show the results
    #print(result)
    #show_result_pyplot(
    #    model,
    #    args.img,
    #    result,
    #    palette=args.palette,
    #    score_thr=args.score_thr,
    #    out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    print(result,"async")
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
