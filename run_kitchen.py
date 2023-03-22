# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from datasets.epic_kitchen import EpicKitchen
from datasets.misc import collate_fn_general, collate_fn_epic_kitchen
from open_vocab_seg.utils import VisualizationDemo
from open_vocab_seg.utils.predictor import OVSegBatchPredictor

from icecream import install
install()

# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset_basedir",
        help="epic-kitchen dataset basedir",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    dataloader = EpicKitchen(cfg, part='P01', clip='P01_01').get_dataloader(batch_size=4,
                                                                            num_workers=4,
                                                                            collate_fn=collate_fn_general,
                                                                            pin_memory=True,
                                                                            shuffle=False)
    predictor = OVSegBatchPredictor(cfg)
    
    for batch_idx, batchdata in enumerate(dataloader):
        # do something with data and labels
        ic(batchdata[0].keys())
        st = time.time()
        predictions = predictor(batchdata)
        ic(time.time() - st)
