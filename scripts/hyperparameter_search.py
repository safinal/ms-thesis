import os
import subprocess
from itertools import product
from tqdm.auto import tqdm
import time

sample_size = [50, 100, 150, 200, 250]
l1_reg = [0, 0.01, 0.1, 1, 0.02, 0.2, 2]
batch_size = [32]
lr = [5e-4]
dataset = 'celeba'
dataset_path = "/home/user01/data/celeba/"
pretrained_path = '/home/user01/dfr-ckpts/celeba/erm_seed1/final_checkpoint.pt'

for count, (s_size, reg, bs, l) in enumerate(tqdm(product(sample_size, l1_reg, batch_size, lr))):
    log_path = f"/home/user01/lfr_logs/{dataset}/s{s_size}_l1{reg}_bs{bs}_lr{l}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(f'{log_path}/result.txt', 'w') as f:
        subprocess.run(['python3', 'main.py',\
                        "--dataset=celeba",\
                        f"--dataset_path={dataset_path}",\
                        "--experiment=loss",\
                        f"--sample_size={s_size}",\
                        f"-b={bs}",\
                        f"-lr={l}",\
                        f"--pretrained_path={pretrained_path}",\
                        "--gamma=0.5",\
                        "--spuriousity=95",\
                        "--weight_decay=1e-4", \
                        "--step_size=85", \
                        f"--l1={reg}",\
                        "--epochs=100",\
                        "--optimizer=adam", \
                        "--seed=1", \
                        "--for_free=True"],stdout=f)

    print(f"finish experiment {count}.")