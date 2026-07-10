import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision

class DominoesMnistCifarDataset(Dataset):
    def __init__(self, data_type, spuriousity):

        assert data_type in ['train', 'last_layer', 'val', 'test'], print("Error! no data_type found")
        assert not data_type in ['val', 'test'] or spuriousity == 0.5, print("Error! val and test must have spuriousity=0.5")
        self.spuriousity = spuriousity
        self.data_type = data_type

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        mydir='path'
        mnist_train_raw = torchvision.datasets.MNIST(f'{mydir}/data/mnist/', train=True, download=True, transform=transform)
        cifar_train_raw = torchvision.datasets.CIFAR10(f'{mydir}/data/cifar10/', train=True, download=True, transform=transform)
        mnist_train, mnist_last_layer, mnist_valid = random_split(mnist_train_raw, [0.64, 0.16, 0.20], generator=torch.Generator().manual_seed(42))
        cifar_train, cifar_last_layer, cifar_valid = random_split(cifar_train_raw, [0.64, 0.16, 0.20], generator=torch.Generator().manual_seed(42))

        mnist_test = torchvision.datasets.MNIST(f'{mydir}/data/mnist/', train=False, download=True, transform=transform)
        cifar_test = torchvision.datasets.CIFAR10(f'{mydir}/data/FashionMNIST/', train=False, download=True, transform=transform)

        mnist_dataset = None
        cifar_dataset = None
        if data_type == 'train':
            mnist_dataset = mnist_train
            cifar_dataset = cifar_train
        elif data_type == 'last_layer':
            mnist_dataset = mnist_last_layer
            cifar_dataset = cifar_last_layer
        elif data_type == 'val':
            mnist_dataset = mnist_valid
            cifar_dataset = cifar_valid
            spuriousity = 0.5
        elif data_type == 'test':
            mnist_dataset = mnist_test
            cifar_dataset = cifar_test


        x, y, g = make_spurious_dataset(mnist_dataset, cifar_dataset, spuriousity)
        self.x = x
        self.y_onehot = nn.functional.one_hot(y.type(torch.LongTensor)).type(torch.FloatTensor)
        self.y = y.tolist()
        self.g = g.tolist()
        self.env_dict = {
            (0., 0.): torch.Tensor(np.array([1,0,0,0])),
            (0., 1.): torch.Tensor(np.array([0,1,0,0])),
            (1., 0.): torch.Tensor(np.array([0,0,1,0])),
            (1., 1.): torch.Tensor(np.array([0,0,0,1]))
        }


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_onehot[idx], self.env_dict[(self.y[idx], self.g[idx])]


def keep_only_lbls(dataset, lbls):
    lbls = {lbl: i for i, lbl in enumerate(lbls)}
    final_X, final_Y = [], []
    for x, y in dataset:
        if y in lbls:
            final_X.append(x)
            final_Y.append(lbls[y])
    X = torch.stack(final_X)
    Y = torch.tensor(final_Y).float().view(-1,1)
    return X, Y

def format_mnist(imgs):
    imgs = np.stack([np.pad(imgs[i][0], 2, constant_values=0)[None,:] for i in range(len(imgs))])
    imgs = np.repeat(imgs, 3, axis=1)
    return torch.tensor(imgs)

def make_spurious_dataset(mnist_dataset, cifar_dataset, spuriousity):
    X_m_0, _ = keep_only_lbls(mnist_dataset, lbls=[0])
    X_m_1, _ = keep_only_lbls(mnist_dataset, lbls=[1])
    X_m_0 = format_mnist(X_m_0.view(-1, 1, 28, 28))
    X_m_1 = format_mnist(X_m_1.view(-1, 1, 28, 28))
    X_m_0 = X_m_0[torch.randperm(len(X_m_0))]
    X_m_1 = X_m_1[torch.randperm(len(X_m_1))]

    X_c_1, _ = keep_only_lbls(cifar_dataset, lbls=[1])
    X_c_9, _ = keep_only_lbls(cifar_dataset, lbls=[9])
    X_c_1 = X_c_1[torch.randperm(len(X_c_1))]
    X_c_9 = X_c_9[torch.randperm(len(X_c_9))]

    min_length = min(len(X_m_0), len(X_m_1), len(X_c_1), len(X_c_9))
    X_m_0, _ = random_split(X_m_0, [min_length, len(X_m_0) - min_length], generator=torch.Generator().manual_seed(42))
    X_m_1, _ = random_split(X_m_1, [min_length, len(X_m_1) - min_length], generator=torch.Generator().manual_seed(42))
    X_c_1, _ = random_split(X_c_1, [min_length, len(X_c_1) - min_length], generator=torch.Generator().manual_seed(42))
    X_c_9, _ = random_split(X_c_9, [min_length, len(X_c_9) - min_length], generator=torch.Generator().manual_seed(42))

    X_m_0_maj, X_m_0_min = random_split(X_m_0, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))
    X_m_1_maj, X_m_1_min = random_split(X_m_1, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))

    X_c_1_maj, X_c_1_min = random_split(X_c_1, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))
    X_c_9_maj, X_c_9_min = random_split(X_c_9, [spuriousity, 1 - spuriousity], generator=torch.Generator().manual_seed(42))

    group_0_X = torch.cat((X_c_1_maj[:], X_m_0_maj[:]), dim=2)
    group_0_Y = torch.zeros(len(group_0_X))
    group_0_G = torch.tensor([0] * len(group_0_X))

    group_1_X = torch.cat((X_c_1_min[:], X_m_1_min[:]), dim=2)
    group_1_Y = torch.zeros(len(group_1_X))
    group_1_G = torch.tensor([1] * len(group_1_X))

    group_2_X = torch.cat((X_c_9_min[:], X_m_0_min[:]), dim=2)
    group_2_Y = torch.ones(len(group_2_X))
    group_2_G = torch.tensor([0] * len(group_2_X))

    group_3_X = torch.cat((X_c_9_maj[:], X_m_1_maj[:]), dim=2)
    group_3_Y = torch.ones(len(group_3_X))
    group_3_G = torch.tensor([1] * len(group_3_X))

    total_x = torch.cat((group_0_X, group_1_X, group_2_X, group_3_X))
    total_y = torch.cat((group_0_Y, group_1_Y, group_2_Y, group_3_Y))
    total_g = torch.cat((group_0_G, group_1_G, group_2_G, group_3_G))

    return total_x, total_y, total_g

def get_dataset(phase, spuriousity):
    assert phase in ['train', 'val', 'test', 'last_layer'], 'Specify a valid split!'
    if spuriousity >= 95 and phase == 'last_layer':
        spuriousity = 95
    dataset = DominoesMnistCifarDataset(data_type=phase, spuriousity=spuriousity)
    return dataset

    
def get_loader(phase, spuriousity=95, **kwargs):
    dataset = get_dataset(phase, spuriousity)
    return DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True if phase == 'train' else False)

def get_dominoes_loaders(spuriousity=95, **kwargs):
    trainloader = get_loader('train', spuriousity=spuriousity/100., batch_size=kwargs.get('batch_size', 128))
    last_layerloader = get_loader('last_layer', spuriousity=spuriousity/100., batch_size=1)
    valloader = get_loader('val', spuriousity=0.5, batch_size=kwargs.get('batch_size', 128))
    testloader = get_loader('test', spuriousity=0.5, batch_size=kwargs.get('batch_size', 128))
    return trainloader, last_layerloader, valloader, testloader