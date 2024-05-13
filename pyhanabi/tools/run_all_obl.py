import os
import sys
import argparse
from easydict import EasyDict as edict
import pprint
pprint = pprint.pprint
import json
import copy
from datetime import datetime
import numpy as np
import random
import subprocess

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

OBL_folder = "icml_OBL"


def run_alll_obl(args):
    jobs = create_jobs(args)
    run_jobs(args, jobs)


def create_jobs(args):
    jobs = []

    for obl_level in range(1,6):
        obl_level_folder = f"{OBL_folder}{obl_level}"
        obl_level_path = os.path.join(args.dir, obl_level_folder)
        obl_model_folders = os.listdir(obl_level_path)
        obl_model_folders.sort()
        if "sweep_r2d2.py" in obl_model_folders:
            obl_model_folders.remove("sweep_r2d2.py")
        if "icml_run3_OBL1" in obl_model_folders:
            obl_model_folders.remove("icml_run3_OBL1")
        for obl_model_folder in obl_model_folders:
            obl_model_path = os.path.join(
                obl_level_path, 
                obl_model_folder,
                "model0.pthw"
            )
            job = edict()
            job.weight1 = obl_model_path
            jobs.append(job)

    return jobs


def run_jobs(args, jobs):
    for job in jobs:
        subprocess.run(["python", "tools/eval_model.py",
            "--weight1", job.weight1,
            "--num_game", str(args.num_game)
        ])
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../models")
    parser.add_argument("--num_game", type=int, default=10000)
    return  parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_alll_obl(args)
