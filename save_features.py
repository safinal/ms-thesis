import torch
import argparse
import os
from tqdm import tqdm

from data import get_waterbirds_loaders, get_celeba_loaders, get_urbancars_loaders, CelebADataset
from utils import get_pretrained_resnet50


def get_dataset_loaders(args):
    '''
        returns trainloader, lastlayer_loader, valloader, testloader with args.batch_size
    '''
    if args.dataset == 'waterbirds':
        return get_waterbirds_loaders(args.dataset_path, batch_size=args.batch_size)
    elif args.dataset == 'celeba':
        return get_celeba_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=4)
    elif args.dataset == 'urbancars':
        return  get_urbancars_loaders(args.dataset_path, args.batch_size, "both")


def get_dataset_loader(args):
    if args.dataset == 'celeba':
        dataset = CelebADataset(phase=None, dataset_dir=args.dataset_path, spuriousity=95, transform='test', sample_size=args.sample_size)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # elif args.dataset == 'waterbirds':
    #     pass
    # elif args.dataset == 'urbancars':
    #     pass
    

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
    parser.add_argument('--dataset', type=str, help='Name of the dataset', required=True, choices=['waterbirds', 'celeba', 'urbancars'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cva', type=bool, default=False)
    parser.add_argument('--sample_size', type=int, default=64)

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_pretrained_resnet50(device, args.pretrained_path, mode='dfr')

    if args.cva:
        sets = {'cva': get_dataset_loader(args)}
    else:
        trainloader, lastlayerloader, valloader, testloader = get_dataset_loaders(args)
        sets = {
            'val': valloader,
            'lastlayer': lastlayerloader,
            'test': testloader,
            'train': trainloader
        }

    if not os.path.exists(os.path.join(args.save_path)):
        os.makedirs(args.save_path)

    model.eval()
    for n, loader in sets.items():
        all_features = []
        all_ys = []
        all_envs = []

        for data in tqdm(loader):
            if len(data) == 3:
                x, y, env = data
            else:
                x, y = data
            with torch.no_grad():
                feature = get_resnet50_embed(model, x.to(device))
            all_features.append(feature.detach().cpu())
            all_ys.append(y)
            if len(data) == 3:
                all_envs.append(env)

        all_features = torch.concat(all_features, 0)
        all_ys = torch.concat(all_ys, 0)
        if all_envs:
            all_envs = torch.concat(all_envs, 0)

        torch.save(all_features,os.path.join(args.save_path, f'{n}_features.pt'))
        torch.save(all_ys,  os.path.join(args.save_path,f'{n}_labels.pt'))
        if not isinstance(all_envs, list):
            torch.save(all_envs, os.path.join(args.save_path,f'{n}_envs.pt'))
