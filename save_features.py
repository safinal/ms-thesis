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
from PIL import Image
import torchvision.transforms as transforms

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
    parser.add_argument('--counterfactual_dir', type=str, default=None, help='Path to counterfactual images directory')
    parser.add_argument('--counterfactual_splits', type=str, default='lastlayer', help='Comma-separated list of splits to apply counterfactuals to (e.g. "lastlayer,val")')

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
    
    # Get test transform for CF images
    if args.dataset == 'waterbirds':
        from data.waterbirds import get_transform_waterbirds
        transform = get_transform_waterbirds(is_training=False)
    elif args.dataset == 'celeba':
        from data.celeba import get_transforms
        transform = get_transforms(is_training=False)
    elif args.dataset == 'urbancars':
        from data.urbancars import get_transforms
        transform = get_transforms("resnet50", is_training=False)
    else:
        transform = None

    for n, loader in sets.items():
        all_features = []
        all_ys = []
        all_envs = []
        global_idx = 0

        for batch, (x, y, env) in enumerate(tqdm(loader)):
            # Replace x with counterfactual images if requested
            if args.counterfactual_dir and n in args.counterfactual_splits.split(','):
                cf_x = []
                for i in range(len(x)):
                    img_path = os.path.join(args.counterfactual_dir, f"cf_{global_idx:06d}.jpg")
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        if transform:
                            img = transform(img)
                        cf_x.append(img)
                    else:
                        print(f"Warning: Missing CF image at {img_path}")
                        cf_x.append(x[i]) # fallback
                    global_idx += 1
                x = torch.stack(cf_x)
            else:
                global_idx += len(x)

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