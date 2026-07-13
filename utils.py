from torch import nn
from models import ResNet50, StdResNet50, ResNet18, LeNet5, bert_pretrained, get_fc
from _test import test_cnn 
import torch 
import torchvision

def weight_init(m):
    """
    Initialize the weights of a given module.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def get_lenet(device, pretrained_path=None):
    model = LeNet5()
    model.device = device
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f'Loaded LeNet-5 from {pretrained_path}')
    return model.to(device)

def get_pretrained_resnet50(device, pretrained_path='100resnet50_erm_ll.model', mode='ours'):
    if mode == 'ours':
        cnn_image_encoder = ResNet50().to(device)
        if pretrained_path != 'imagenet':
            print(f'loaded {pretrained_path} as model')
            print(cnn_image_encoder.load_state_dict(torch.load(pretrained_path, map_location=device)))
        return cnn_image_encoder
    elif mode == 'dfr':
        n_classes = 2
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
        model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        d = model.fc.in_features
        model.fc = torch.nn.Linear(d, n_classes).to(device)
        if pretrained_path != 'imagenet':
            checkpoint = torch.load(pretrained_path) if torch.cuda.is_available() else torch.load(pretrained_path, map_location=torch.device('cpu'))
            try:
                checkpoint = checkpoint['classifier']
            except:
                pass
            model.load_state_dict(checkpoint)
            if torch.cuda.is_available():
                model = model.cuda()
                model.device = "cuda"
            print(f'loaded {pretrained_path}')
        return model

def get_pretrained_resnet18(device, pretrained_path='100resnet50_erm_ll.model', mode='dfr'):
    if mode == 'dfr':
        n_classes = 2
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        d = model.fc.in_features
        model.fc = torch.nn.Linear(d, n_classes)
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint)
        model = model.cuda()
        model.device = "cuda"
        print(f'loaded {pretrained_path}')
        return model
    cnn_image_encoder = ResNet18().to(device)
    if pretrained_path != 'random':
        print(f'loaded {pretrained_path} as model')
        print(cnn_image_encoder.load_state_dict(torch.load(pretrained_path, map_location=device)))
    return cnn_image_encoder

def get_pretrained_stdresnet(pretrained_path='100resnet50_erm_ll.model'):
    return StdResNet50(2, pretrained_path)

def get_pretrained_bert(pretrained_path, num_classes=2, device=torch.device('cuda')):
    model = bert_pretrained(num_classes).to(device)

    model.load_state_dict(torch.load(pretrained_path))
    return model

def eval_model(trainloader, valloader, testloader, model, lastlayerloader=None, args=None):
    print('Train:', '-'*50, sep='\n')
    avg_inv_acc, worst_acc, miscls_envs, corrcls_envs = test_cnn(trainloader, model, args=args, inferred_groups=False, return_samples=True)
    torch.save(miscls_envs[3], 'wtf.pt')
    if lastlayerloader:
        print('-'*50, 'LastLayer:', '-'*50, sep='\n')
        test_cnn(lastlayerloader, model, args=args, inferred_groups=False)
    print('-'*50, 'Validation:', '-'*50, sep='\n')
    test_cnn(valloader, model, args=args, inferred_groups=False)
    print('-'*50, 'Test:', '-'*50, sep='\n')
    test_cnn(testloader, model, args=args, inferred_groups=False)

def save_model(model, path='unknown.model'):
    torch.save(model.state_dict(), path)