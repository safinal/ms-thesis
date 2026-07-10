import sys
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import exps
import data
import utils
import _test as test
import run
import mlflow
import os
import models
import random
import numpy as np
import json
import copy
from train import *
from _test import *
import numpy as np
from data.feature_dataset import get_feature_loader

def generate_optimizer_and_scheduler(model, learning_rate, step_size, gamma, optimizer_type, l2=0):
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    elif optimizer_type == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2)
    else:
        raise ValueError("Invalid optimizer type. Supported options are 'adam', 'adamW', and 'SGD'.")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5)
    return optimizer, scheduler


def get_dataset_loaders(args):
    '''
        returns trainloader, lastlayer_loader, valloader, testloader with args.batch_size
    '''
    if args.feature_only:
        if args.validation_path:
            print ('Loading validation data from the provided path.')
            return data.get_feature_loaders(args.dataset_path, args.batch_size, validation_path = args.validation_path)
        else:
            return data.get_feature_loaders(args.dataset_path, args.batch_size)

    elif args.dataset == 'waterbirds':
        return data.get_waterbirds_loaders(args.dataset_path, batch_size=args.batch_size)
    elif args.dataset == 'celeba':
        return data.get_celeba_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=1)
    elif args.dataset == 'civilcomments':
        return data.get_civil_comments_loaders(args.pretrained_path, args.dataset_path, args.batch_size)
    elif args.dataset == 'multinli':
        return data.get_multinli_loaders(args.dataset_path, batch_size=16, num_workers=2)
    elif args.dataset == 'urbancars':
        return data.get_urbancars_loaders(args.dataset_path, args.batch_size, "both")



def freeze_model(model, reinit = True):
    # ret = copy.deepcopy(model)
    if hasattr(model, "model"):
        if reinit:
            utils.weight_init(model.model.fc)
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.fc.parameters():
            param.requires_grad = True
    else:
        if reinit:
            utils.weight_init(model.fc)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    print('Last fc layer has been re-initialized successfully!')
    print('Model freezed! Have fun with your last layer experiment')
    return model


def generate_experiment(args, model=None):
    if args.experiment == 'DFR':
        return exps.DFR()
    elif args.experiment == 'loss':
        return exps.LossBasedExp()
    elif args.experiment == 'cluster':
        return exps.ClusterBasedExp()
    elif args.experiment == 'entropy':
        return exps.EntropyBasedExp()
    elif args.experiment == 'gradcam':
        return exps.GradCAMExp(model)


def train_early_stop(model, trainloader, valloader):
    optimizer, scheduler = generate_optimizer_and_scheduler(model, 0.00001, 10, 0.5, 'adam', l2=0)
    for i in range (np.random.randint(1,3)):
        train_cnn(trainloader, model, optimizer, scheduler, i, torch.device('cuda'), 0,  log = False)
        # acc, _ = test_cnn(valloader, model, log=False, args=args, inferred_groups=False)


def get_early_stop_valloaders(model, args, trainloader, valloader, path):
    valloaders = []
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range (args.num_val):
        save_path = path + '/val' + str(i) + '.pt'
        if os.path.exists(save_path):
            val_model = freeze_model(model, reinit=False)
            val_model.load_state_dict(torch.load(save_path))
        else:
            val_model = freeze_model(model, reinit=False)
            train_early_stop(val_model, trainloader, valloader)
            torch.save (val_model.state_dict(), save_path)

        _, _, miscls_envs, corrcls_envs = test.test_cnn(valloader, val_model, return_samples=True,
                                                                      args=args)
        new_valloader = experiment.create_balanced_dataloader_val(miscls_envs, corrcls_envs,
                                                                     sample_size=99999999999,
                                                                     model=val_model, batch_size=valloader.batch_size,
                                                                     for_free=args.for_free)

        print('validation labels:', new_valloader.dataset.tensors[1].argmax(1).unique(return_counts=True), sep='\n')
        print('validation groups:', new_valloader.dataset.tensors[2].argmax(1).unique(return_counts=True), sep='\n')

        valloaders.append(new_valloader)

    return valloaders



def get_cls_valloaders (model, args, valloader):
    valloaders = []

    # save_dir = args.validation_path
    #
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    model.eval()
    for i in range (args.num_val):
        reinit = True
        if args.error_splitting:
            reinit = False
        ret = freeze_model(model, reinit=reinit)
        avg_acc, worst_acc, miscls_envs, corrcls_envs = test.test_cnn(valloader, ret, return_samples=True,
                                                                      args=args)
        for g in range(n_envs):
            print(f'for env{g}:\n\tmiscls:', end=' ')
            print(len(miscls_envs[g]))
            print('\tcorrcls:', end=' ')
            print(len(corrcls_envs[g]))
        if not args.random_grouping:
            random_valloader = experiment.create_balanced_dataloader_val(miscls_envs, corrcls_envs, sample_size=99999999999,
                                                                  model=ret, batch_size=valloader.batch_size,
                                                                  for_free=args.for_free)
        else:
            random_valloader = experiment.create_balanced_random_dataloader({0: miscls_envs[0] + miscls_envs[1] +
                                                                         corrcls_envs[0] + corrcls_envs[1],
                                                                      1: miscls_envs[2] + miscls_envs[3] +
                                                                         corrcls_envs[2] + corrcls_envs[3]},
                                                                     batch_size=valloader.batch_size)
        print('validation labels:', random_valloader.dataset.tensors[1].argmax(1).unique(return_counts=True), sep='\n')
        print('validation groups:', random_valloader.dataset.tensors[2].argmax(1).unique(return_counts=True), sep='\n')

        valloaders.append(random_valloader)

    return valloaders

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spurious Correlation Experiment')
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer',
                        choices=['adam', 'adamW', 'SGD'])
    parser.add_argument('--experiment', type=str, help='Type of experiment',
                        choices=['ERM', 'DFR', 'loss', 'cluster', 'entropy', 'gradcam'])
    parser.add_argument('--dataset', type=str, default='waterbirds',
                        help='Name of the dataset',
                        choices=['waterbirds', 'celeba', 'multinli', 'civilcomments', 'urbancars'],
                        required=True)
    parser.add_argument('--dataset_path', type=str, default='./waterbird_complete_forest2water2',
                        help='Path of the dataset')
    parser.add_argument('--comments', type=str, default='',
                        help='comments to be included in the name of logs')
    parser.add_argument('--output_path', type=str, default='./', help='Path of the logs and checkpoints')
    parser.add_argument('--bert_ckpt', type=str, default='bert-base-uncased',
                        help='weights of pretrained bert for tokenization')
    parser.add_argument('--sample_size', type=int, default=64, help='Sample size of each group in the experiment')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coefficient for L2 regularization')
    parser.add_argument('--l1', type=float, default=0, help='Weight decay coefficient for L1 regularization')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--model', type=str, default='resnet', help='Name of the model to use',
                        choices=['ResNet', 'BERT'])
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path of the .model file')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for last layer re-training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of CPU cores to use')
    parser.add_argument('--test_only', type=bool, default=False, help='Just test the specified model on the dataset')
    parser.add_argument('--log', type=bool, default=True, help='Whether log the experiment on wandb or not')
    parser.add_argument('--for_free', type=bool, default=False,
                        help='choose the best model based on group-inferred validation data')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--random_grouping', type=bool, default=False, help='randomly group validation data')
    parser.add_argument('--feature_only', type=bool, default=False, help='load features instead of the raw data')
    parser.add_argument('--num_val', type=int, default=1, help='number of validation sets')
    parser.add_argument('--fine_tune', type=bool, default=False, help='fine-tune the classifier')
    parser.add_argument('--early_stop_val', type=bool, default=False, help='use early-stop models for validation grouping')
    parser.add_argument('--validation_path', type=str, default=None, help='Path to validation grouping models')
    parser.add_argument('--saved_val', type=bool, default=False, help='use saved validation set.')
    parser.add_argument('--error_splitting', type=bool, default=False, help='use error splitting for environment inference.')

    args = parser.parse_args()

    save_dir = os.path.join(args.output_path,
                            f"{args.experiment}_{args.comments}_{args.dataset}_LR{args.learning_rate}_step{args.step_size}_gamma{args.gamma}_seed{args.seed}_samples{args.sample_size}_l1{args.l1}/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args_dict = vars(args)

    print(json.dumps(args_dict, indent=4))

    ############ SEED #################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    ###################################################
    trainloader, lastlayerloader, valloader, testloader = get_dataset_loaders(args)

    n_envs = data.dataset_specs.datasets[args.dataset]['num_envs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.feature_only:
        n = data.dataset_specs.datasets[args.dataset]['num_classes']
        d = data.dataset_specs.datasets[args.dataset]['hidden_layer_size']
        model = utils.get_fc(device, args.pretrained_path, num_features = d, num_classes=n)
    elif args.dataset == 'civilcomments':
        model = utils.get_pretrained_bert(args.pretrained_path, 2, device)
    elif args.dataset == 'multinli':
        model = utils.get_pretrained_bert(args.pretrained_path, 3, device)
    else:
        model = utils.get_pretrained_resnet50(device, args.pretrained_path, mode='dfr')

    if args.test_only:
        model.zero_grad()
        with torch.no_grad():
            utils.eval_model(trainloader, valloader, testloader, model, lastlayerloader=lastlayerloader, args=args)
    else:
        if args.experiment != 'ERM':
            print ('Accuracy of ERM on the test set')
            _,_ = test.test_cnn(testloader, model, return_samples=False, args=args, inferred_groups=False)

            # model = freeze_model(model) # Uncomment if you want to infer lastlayer based on random classifier
            experiment = generate_experiment(args, model)
            avg_acc, worst_acc, miscls_envs, corrcls_envs = test.test_cnn(lastlayerloader, model, return_samples=True,
                                                                          args=args)
            for g in range(4):
                print(f'for env{g}:\n\tmiscls:', end=' ')
                print(len(miscls_envs[g]))
                print('\tcorrcls:', end=' ')
                print(len(corrcls_envs[g]))
            balanced_loader = experiment.create_balanced_dataloader_ll(miscls_envs, corrcls_envs,
                                                                       sample_size=args.sample_size,
                                                                       model=model, batch_size=args.batch_size,
                                                                       dataloader=lastlayerloader, dataset=args.dataset)
            print('lastlayer labels:', balanced_loader.dataset.tensors[1].argmax(1).unique(return_counts=True),
                  sep='\n')
            print('lastlayer groups:', balanced_loader.dataset.tensors[2].argmax(1).unique(return_counts=True),
                  sep='\n')

        if args.for_free:
            ############ SEED ################################# Uncomment if you want to change seed in this stage
            # torch.manual_seed(args.seed+40)
            # torch.cuda.manual_seed(args.seed+40)
            # torch.backends.cudnn.deterministic = True
            # random.seed(args.seed+40)
            # np.random.seed(args.seed+40)
            # os.environ['PYTHONHASHSEED'] = str(args.seed+40)
            ###################################################

            print(f'Enjoy for free mode!')
            experiment = generate_experiment(args, model)

            if args.early_stop_val:
                valloaders = get_early_stop_valloaders(model, args, lastlayerloader, valloader, args.validation_path)

            else:
                valloaders = [valloader]

        optimizer, scheduler = generate_optimizer_and_scheduler(model, args.learning_rate, args.step_size,
                                                                args.gamma, args.optimizer, args.weight_decay)
        
        valloaders = [valloader]
        if args.experiment != 'ERM':
            if args.fine_tune:
                model = freeze_model(model, reinit=False)
            else:
                model = freeze_model(model, reinit=True)

            result = run.run_last_layer_experiment(model, device, balanced_loader, valloaders,
                                                   args.experiment,
                                                   optimizer, args.l1, scheduler, dataset=args.dataset,
                                                   epochs=args.epochs, seed=args.seed, args=args)
        else:
            result = run.run_last_layer_experiment(model, device, trainloader, valloaders,
                                                       args.experiment,
                                                       optimizer, args.l1, scheduler, dataset=args.dataset,
                                                       epochs=args.epochs, seed=args.seed, args=args)
        print(f'Best model saved at {result}')

        if args.feature_only:
            n = data.dataset_specs.datasets[args.dataset]['num_classes']
            d = data.dataset_specs.datasets[args.dataset]['hidden_layer_size']
            model.fc = torch.nn.Linear(d, n)
            checkpoint = torch.load(result)
            model.load_state_dict(checkpoint)
            test_model = model.cuda()
            test_model.device = "cuda"

        elif args.dataset=='civilcomments' or args.dataset=='multinli':
            test_model = utils.get_pretrained_bert(result)

        else:
            n_classes = data.dataset_specs.datasets[args.dataset]['num_classes']
            model = torchvision.models.resnet50(weights=None)
            d = model.fc.in_features
            model.fc = torch.nn.Linear(d, n_classes)
            checkpoint = torch.load(result)
            model.load_state_dict(checkpoint)
            test_model = model.cuda()
            test_model.device = "cuda"

        if args.for_free:
            val_avg, val_worst = run.multi_eval(test_model, valloaders, False, args)
        else:
            val_avg, val_worst = test.test_cnn(valloader, test_model, return_samples=False, args=args, inferred_groups=True) # TODO

        test_avg, test_worst = test.test_cnn(testloader, test_model, return_samples=False, args=args, inferred_groups=False)

        res_dict = {'val':{'avg': val_avg, 'worst':val_worst}, 'test': {'avg': test_avg , 'worst':test_worst}}
        print (res_dict)
        print(f'Best model saved at {result}')
        res_dict['config'] = args_dict
        json.dump(res_dict, open(os.path.join(save_dir, "results.json"), 'w'))

        print('Execution Finished')
        sys.exit(1)