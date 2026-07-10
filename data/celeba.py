import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


class celebADataset(Dataset):
    def __init__(self, phase, dataset_dir, spuriousity, transform = "train"):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
            'last_layer': 3,
        }
        # (y, gender)
        self.env_dict = {
            (0, 0): torch.Tensor([1, 0, 0, 0]).type(torch.LongTensor),   # nonblond hair, female
            (0, 1): torch.Tensor([0, 1, 0, 0]).type(torch.LongTensor),   # nonblond hair, male
            (1, 0): torch.Tensor([0, 0, 1, 0]).type(torch.LongTensor),   # blond hair, female
            (1, 1): torch.Tensor([0, 0, 0, 1]).type(torch.LongTensor)    # blond hair, male
        }
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celeba_metadata.csv')) # f'balanced_celeba_split{spuriousity}_with_additional_last_layer.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[phase]]

        self.y_array = self.metadata_df['y'].values
        self.gender_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        if transform=='train':
            transform = transforms.Compose(
                                [transforms.ToTensor(), 
                                 transforms.RandomResizedCrop(
                                    (224, 224),
                                    scale=(0.7, 1.0),
                                    ratio=(0.75, 1.3333333333333333),
                                    interpolation=2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])] 
                             )
        else:
            transform = transforms.Compose(
                            [transforms.Resize((256, 256)),
                             transforms.CenterCrop((224, 224)),
                             transforms.ToTensor(), 
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])]
                        )
        self.transform = transform
        print(torch.Tensor(self.y_array).type(torch.LongTensor).shape)
        self.labels = torch.nn.functional.one_hot(torch.Tensor(self.y_array).type(torch.LongTensor)).type(torch.FloatTensor)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        env = self.env_dict[(y, gender)]

        return img, label, env

    def get_images(self):
        print('Generating CelebA...')
        mean = (0, 0, 0)
        std = (1, 1, 1)
        trans = transforms.Compose([
                # transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])
        images = []
        for idx in tqdm(range(len(self.filename_array))):
            img_filename = os.path.join(
                self.dataset_dir,
                'img_align_celeba',
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            images.append(torch.unsqueeze(trans(img), 0))
        return torch.cat(images, dim=0)

    def get_envs(self):
        envs = []
        for idx in tqdm(range(len(self.filename_array))):
            y = self.y_array[idx]
            gender = self.gender_array[idx]
            env = self.env_dict[(y, gender)]
            envs.append(torch.unsqueeze(env, 0))
        return torch.cat(envs, dim=0)



def get_dataset (phase, dataset_dir, spuriousity=95, transform = "train"):
    dataset = celebADataset(phase=phase, dataset_dir=dataset_dir, spuriousity=spuriousity, transform=transform)
    return dataset

def get_loader (phase, spuriousity=95, transform = "train", **kwargs):
    assert 'data_path' in kwargs.keys(),'`data_path` argument missing!'
    dataset = get_dataset(phase, kwargs['data_path'], spuriousity, transform)
    return DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=kwargs['num_workers'])

def get_celeba_loaders(data_path='./', spuriousity=95, **kwargs):
    trainloader = get_loader('train', spuriousity=spuriousity, data_path=data_path, batch_size=kwargs.get('batch_size', 512), num_workers=kwargs.get('num_workers', 1), transform='train')
    last_layerloader = get_loader('last_layer', spuriousity=spuriousity, data_path=data_path, batch_size=kwargs.get('batch_size', 512), num_workers=kwargs.get('num_workers', 1), transform='test')
    valloader = get_loader('val', spuriousity=spuriousity, data_path=data_path, batch_size=kwargs.get('batch_size', 512), num_workers=kwargs.get('num_workers', 1), transform='test')
    testloader = get_loader('test', spuriousity=spuriousity, data_path=data_path, batch_size=512, num_workers=kwargs.get('num_workers', 1), transform='test')
    return trainloader, last_layerloader, valloader, testloader