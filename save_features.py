import utils
import data
import torch
from spuco.group_inference import EIIL
import torch
import models
import data
import utils
import argparse
from tqdm import tqdm
import os
import random
import numpy as np
from torchvision.models import resnet18
import torch.nn as nn

def get_dataset_loaders(args):
    '''
        returns trainloader, lastlayer_loader, valloader, testloader with args.batch_size
    '''
    if args.dataset == 'waterbirds':
        return data.get_waterbirds_loaders(args.dataset_path, batch_size=args.batch_size)
    elif args.dataset == 'celeba':
        return data.get_celeba_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=4)
    elif args.dataset == 'urbancars':
        return  data.get_urbancars_loaders(args.dataset_path, args.batch_size, "both")


#source: https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/dfr_evaluate_spurious.py
def get_resnet50_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature extraction')
    parser.add_argument('--dataset', type=str, default='waterbirds',
                        help='Name of the dataset',
                        choices=['waterbirds', 'celeba', 'urbancars'],
                        required=True)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()



    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = utils.get_pretrained_resnet50(device, args.pretrained_path, mode='dfr')

    trainloader, lastlayerloader, valloader, testloader = get_dataset_loaders(args)
    sets = {
            'val': valloader,
            'lastlayer': lastlayerloader,
            'test':testloader,
            'train':trainloader
            }

    if not os.path.exists(os.path.join(args.save_path)):
        os.makedirs(args.save_path)

    model.eval()
    for n, loader in sets.items():
        all_features = []
        all_ys = []
        all_envs = []

        for batch, (x, y, env) in enumerate(tqdm(loader)):
            with torch.no_grad():
                feature = get_resnet50_embed(model, x.to(device))
            all_features.append(feature.detach().cpu())
            all_ys.append(y)
            all_envs.append(env)

        all_features = torch.concat(all_features, 0)
        all_ys = torch.concat(all_ys, 0)
        all_envs = torch.concat(all_envs, 0)

        print (all_features.shape, all_ys.shape, all_envs.shape)

        torch.save (all_features,os.path.join(args.save_path, f'{n}_features.pt'))
        torch.save(all_ys,  os.path.join(args.save_path,f'{n}_labels.pt'))
        torch.save(all_envs, os.path.join(args.save_path,f'{n}_envs.pt'))