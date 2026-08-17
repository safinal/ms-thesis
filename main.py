import argparse
import numpy as np
import os
import random
import json
import torch
import torchvision

from exps import DFR, LossBasedExp, ClusterBasedExp, CVA, EntropyBasedExp
from data import get_feature_loaders, get_urbancars_loaders, get_celeba_loaders, get_waterbirds_loaders, dataset_specs
from train import train_cnn
from _test import test_cnn
from run import run_last_layer_experiment, multi_eval
from utils import weight_init, get_fc, eval_model, get_pretrained_resnet50


def generate_optimizer_and_scheduler(model, learning_rate, step_size, gamma, optimizer_type, l2=0):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    elif optimizer_type == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2)
    else:
        raise ValueError("Invalid optimizer type. Supported options are 'adam', 'adamW', and 'SGD'.")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


def get_dataset_loaders(args):
    '''
        returns trainloader, lastlayer_loader, valloader, testloader with args.batch_size
    '''
    if args.feature_only:
        if args.validation_path:
            print ('Loading validation data from the provided path.')
            return get_feature_loaders(args.dataset_path, args.batch_size, validation_path=args.validation_path)
        else:
            return get_feature_loaders(args.dataset_path, args.batch_size)

    elif args.dataset == 'waterbirds':
        return get_waterbirds_loaders(args.dataset_path, batch_size=args.batch_size)
    elif args.dataset == 'celeba':
        return get_celeba_loaders(args.dataset_path, batch_size=args.batch_size, num_workers=1)
    elif args.dataset == 'urbancars':
        return get_urbancars_loaders(args.dataset_path, args.batch_size, "both")



def freeze_model(model, reinit = True):
    # ret = copy.deepcopy(model)
    if hasattr(model, "model"):
        if reinit:
            weight_init(model.model.fc)
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.fc.parameters():
            param.requires_grad = True
    else:
        if reinit:
            weight_init(model.fc)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    print('Last fc layer has been re-initialized successfully!')
    print('Model freezed! Have fun with your last layer experiment')
    return model


def generate_experiment(args, model=None):
    if args.experiment == 'DFR':
        return DFR()
    elif args.experiment == 'loss':
        return LossBasedExp()
    elif args.experiment == 'cluster':
        return ClusterBasedExp()
    elif args.experiment == 'entropy':
        return EntropyBasedExp()
    # elif args.experiment == 'gradcam':
    #     return GradCAMExp(model)
    elif args.experiment == 'CVA':
        return CVA()



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

        _, _, miscls_envs, corrcls_envs = test_cnn(valloader, val_model, return_samples=True, args=args)
        new_valloader = experiment.create_balanced_dataloader_val(
            miscls_envs, corrcls_envs,
            sample_size=99999999999,
            model=val_model, 
            batch_size=valloader.batch_size,
            for_free=args.for_free
        )

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
        avg_acc, worst_acc, miscls_envs, corrcls_envs = test_cnn(valloader, ret, return_samples=True, args=args)
        for g in range(n_envs):
            print(f'for env{g}:\n\tmiscls:', end=' ')
            print(len(miscls_envs[g]))
            print('\tcorrcls:', end=' ')
            print(len(corrcls_envs[g]))
        if not args.random_grouping:
            random_valloader = experiment.create_balanced_dataloader_val(
                miscls_envs, 
                corrcls_envs, 
                sample_size=99999999999,
                model=ret, 
                batch_size=valloader.batch_size,
                for_free=args.for_free
            )
        else:
            random_valloader = experiment.create_balanced_random_dataloader(
                {
                    0: miscls_envs[0] + miscls_envs[1] + corrcls_envs[0] + corrcls_envs[1],
                    1: miscls_envs[2] + miscls_envs[3] + corrcls_envs[2] + corrcls_envs[3]
                },
                batch_size=valloader.batch_size
            )
        print('validation labels:', random_valloader.dataset.tensors[1].argmax(1).unique(return_counts=True), sep='\n')
        print('validation groups:', random_valloader.dataset.tensors[2].argmax(1).unique(return_counts=True), sep='\n')

        valloaders.append(random_valloader)

    return valloaders

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spurious Correlation Experiment')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamW', 'SGD'])
    parser.add_argument('--experiment', type=str, required=True, choices=['ERM', 'DFR', 'loss', 'cluster', 'entropy', 'gradcam', 'CVA'])
    parser.add_argument('--dataset', type=str, required=True, choices=['waterbirds', 'celeba', 'urbancars'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--comments', type=str, default='', help='comments to be included in the name of logs')
    parser.add_argument('--output_path', type=str, required=True, help='Path of the logs and checkpoints')
    parser.add_argument('--sample_size', type=int, default=64, help='Sample size of each group in the experiment')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coefficient for L2 regularization')
    parser.add_argument('--l1', type=float, default=0, help='Weight decay coefficient for L1 regularization')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path of the pretrained model file')
    parser.add_argument('--ba'tch_size, type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU cores to use')
    parser.add_argument('--test_only', type=bool, default=False, help='Just test the specified model on the dataset')
    parser.add_argument('--log', type=bool, default=True, help='Whether log the experiment on wandb or not')
    parser.add_argument('--for_free', type=bool, default=False, help='choose the best model based on group-inferred validation data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_grouping', type=bool, default=False, help='randomly group validation data')
    parser.add_argument('--feature_only', type=bool, default=False, help='load features instead of the raw data')
    parser.add_argument('--num_val', type=int, default=1, help='number of validation sets')
    parser.add_argument('--fine_tune', type=bool, default=False, help='fine-tune the classifier')
    parser.add_argument('--early_stop_val', type=bool, default=False, help='use early-stop models for validation grouping')
    parser.add_argument('--validation_path', type=str, default=None, help='Path to validation grouping models')
    parser.add_argument('--saved_val', type=bool, default=False, help='use saved validation set.')
    parser.add_argument('--error_splitting', type=bool, default=False, help='use error splitting for environment inference.')
    parser.add_argument('--balanced_dataset_path', type=str, default=None)


    args = parser.parse_args()
    
    editing_model_name = args.balanced_dataset_path.split('/')[-2] if args.balanced_dataset_path is not None else None
    save_dir = os.path.join(
        args.output_path,
        f"{args.experiment}_{args.comments}_{args.dataset}_opt-{args.optimizer}_batchsize{args.batch_size}_LR{args.learning_rate}_step{args.step_size}_gamma{args.gamma}_seed{args.seed}_samples{args.sample_size}_l1{args.l1}_feature-{args.feature_only}_editing-{editing_model_name}/"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args_dict = vars(args)

    print(json.dumps(args_dict, indent=4))

    # os.environ["WANDB_DIR"] = './'
    # os.environ["WANDB_CONFIG_DIR"] = './wandb/config/'
    # os.environ["WANDB_CACHE_DIR"] = './wandb/cache/'
    # os.environ["WANDB_DATA_DIR"] = './wandb/data/'

    ############ SEED #################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    ###################################################
    trainloader, lastlayerloader, valloader, testloader = get_dataset_loaders(args)

    n_envs = dataset_specs.datasets[args.dataset]['num_envs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.feature_only:
        n = dataset_specs.datasets[args.dataset]['num_classes']
        d = dataset_specs.datasets[args.dataset]['hidden_layer_size']
        model = get_fc(device, args.pretrained_path, num_features=d, num_classes=n)
    else:
        model = get_pretrained_resnet50(device, args.pretrained_path, mode='dfr')

    if args.test_only:
        model.zero_grad()
        with torch.no_grad():
            eval_model(trainloader, valloader, testloader, model, lastlayerloader=lastlayerloader, args=args)
    else:
        if args.experiment != 'ERM':
            print('Accuracy of ERM on the test set')
            _, _ = test_cnn(testloader, model, return_samples=False, args=args, inferred_groups=False)
            
            # model = freeze_model(model) # Uncomment if you want to infer lastlayer based on random classifier
            experiment = generate_experiment(args, model)
            avg_acc, worst_acc, miscls_envs, corrcls_envs = test_cnn(lastlayerloader, model, return_samples=True, args=args)
            for g in range(4):
                print(f'for env{g}:\n\tmiscls:', end=' ')
                print(len(miscls_envs[g]))
                print('\tcorrcls:', end=' ')
                print(len(corrcls_envs[g]))
            balanced_loader = experiment.create_balanced_dataloader_ll(
                miscls_envs=miscls_envs, 
                corrcls_envs=corrcls_envs,
                sample_size=args.sample_size,
                model=model, 
                batch_size=args.batch_size,
                dataloader=lastlayerloader, 
                dataset=args.dataset, 
                balanced_dataset_path=args.balanced_dataset_path,
                feature_only=args.feature_only
            )
            # print('lastlayer labels:', balanced_loader.dataset.tensors[1].argmax(1).unique(return_counts=True), sep='\n')
            # print('lastlayer groups:', balanced_loader.dataset.tensors[2].argmax(1).unique(return_counts=True), sep='\n')

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

        optimizer, scheduler = generate_optimizer_and_scheduler(
            model, 
            args.learning_rate, 
            args.step_size,
            args.gamma, 
            args.optimizer, 
            args.weight_decay
        )
        
        valloaders = [valloader]
        if args.experiment != 'ERM':
            if args.fine_tune:
                model = freeze_model(model, reinit=False)
            else:
                model = freeze_model(model, reinit=True)

            result = run_last_layer_experiment(
                model, 
                device, 
                balanced_loader, 
                valloaders,
                optimizer, 
                args.l1, 
                scheduler, 
                epochs=args.epochs, 
                args=args,
                save_dir=save_dir
            )
        else:
            result = run_last_layer_experiment(
                model, 
                device, 
                trainloader, 
                valloaders,
                optimizer, 
                args.l1, 
                scheduler, 
                epochs=args.epochs, 
                args=args,
                save_dir=save_dir
            )
        print(f'Best model saved at {result}')

        if args.feature_only:
            n = dataset_specs.datasets[args.dataset]['num_classes']
            d = dataset_specs.datasets[args.dataset]['hidden_layer_size']
            model.fc = torch.nn.Linear(d, n)
            checkpoint = torch.load(result)
            model.load_state_dict(checkpoint)
            test_model = model.cuda()
            test_model.device = "cuda"

        else:
            n_classes = dataset_specs.datasets[args.dataset]['num_classes']
            model = torchvision.models.resnet50(weights=None)
            d = model.fc.in_features
            model.fc = torch.nn.Linear(d, n_classes)
            checkpoint = torch.load(result)
            model.load_state_dict(checkpoint)
            test_model = model.cuda()
            test_model.device = "cuda"

        if args.for_free:
            val_avg, val_worst = multi_eval(test_model, valloaders, False, args)
        else:
            val_avg, val_worst = test_cnn(valloader, test_model, return_samples=False, args=args, inferred_groups=True)

        test_avg, test_worst = test_cnn(testloader, test_model, return_samples=False, args=args, inferred_groups=False)

        res_dict = {'val':{'avg': val_avg, 'worst':val_worst}, 'test': {'avg': test_avg , 'worst':test_worst}}
        print (res_dict)
        print(f'Best model saved at {result}')
        res_dict['config'] = args_dict
        json.dump(res_dict, open(os.path.join(save_dir, "results.json"), 'w'))
        print('Execution Finished')