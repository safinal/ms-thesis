import os
import subprocess
from itertools import product
from tqdm.auto import tqdm
import time

sample_size = [10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300]
l1_reg = [0, 0.01, 0.05, 0.1, 0.2, 1, 2]
batch_size = [32]
lr = [1e-4, 1e-3]
weight_decays = [0]
scheduler_step_size = [85]
dataset = 'celeba'
dataset_path = '/path/to/data/celeba/features_noaug_seed1'
pretrained_path = '/path/to/dfr-ckpts/celeba/erm_seed1/final_checkpoint.pt'
comments = ''
log_path = './celeba_LFR_finetuned.sh'
with open(log_path, 'w') as f:
    for count, (s_size, reg, bs, l, step_size, wd) in enumerate(product(sample_size, l1_reg, batch_size, lr, scheduler_step_size, weight_decays)):
            print(' '.join(['python', 'main.py',\
                            f"--dataset {dataset}",\
                            f"--dataset_path {dataset_path}",\
                            "--experiment loss",\
                            f"--sample_size {s_size}",\
                            f"-b {bs}",\
                            f"-lr {l}",\
                            f"--pretrained_path {pretrained_path}",\
                            f"--gamma 0.1",\
                            f"--weight_decay {wd}",\
                            f"--l1 {reg}",\
                            f"--epochs 100",\
                            f"--optimizer adam",\
                            f"--step_size {step_size}",\
                            f"--seed 1",\
                            # f"--output_path {output_path}",\
                            # "--for_free True",\
                            f"--fine_tune True",\
                            # f"--validation_path {validation_path}",\
                            "--feature_only True"]), file = f, flush=True)
