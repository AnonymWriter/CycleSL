import copy
import torch
import random
import argparse
import numpy as np
from datetime import datetime

# self-defined functions
from models import *
from server import Server
from data_preprocessing import *

def seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
def Args() -> argparse.Namespace:
    """
    Helper function for argument parsing.

    Returns:
        args (argparse.Namespace): parsed argument object.
    """

    parser = argparse.ArgumentParser()
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'femnist', help = 'project name, from femnist, celeba, shakespeare, covid19')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-SL', '--switch_SL', type = str, default = 'SSL', help = 'SL algorithm')
    parser.add_argument('--fast_eval', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to skip evaluation for training datasets to speed up experiments')
    
    # hyperparameters
    parser.add_argument('-L', '--cut_layer', type = int, default = -1, help = 'cut layer. -1 stands for using default cut layer')
    parser.add_argument('-T', '--num_round', type = int, default = 601, help = 'number of split learning rounds')
    parser.add_argument('-E', '--num_server_epoch', type = int, default = 1, help = 'number of server epochs (only for CycleSL-based methods)')
    parser.add_argument('-K', '--keep_rate', type = float, default = 1.0, help = 'proportion of clients considered. Useful when the dataset contains too many clients.')
    parser.add_argument('-C', '--client_C', type = float, default = 0.05, help = 'number / proportion of participating clients in each aggregation round')
    parser.add_argument('-S', '--min_sample', type = int, default = 32, help = 'minimal required amount of samples per client')
    parser.add_argument('-c_bs', '--client_bs', type = int, default = 32, help = 'batch size for client training data loader')
    parser.add_argument('-s_bs', '--server_bs', type = int, default = 32, help = 'batch size for server training data loader')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = 3e-4, help = 'client learning rate')
    parser.add_argument('-s_lr', '--server_lr', type = float, default = 3e-4, help = 'server learning rate')
    parser.add_argument('-c_op', '--client_optim', type = str, default = 'Adam', help = 'client optimizer')
    parser.add_argument('-s_op', '--server_optim', type = str, default = 'Adam', help = 'server optimizer')
    parser.add_argument('--default', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use default hyperparmeters')
    parser.add_argument('-resnet', '--resnet', type = int, default = 9, help = 'whether to use resnet9 or resnet34. Only for CIFAR dataset.')

    # path parameters
    parser.add_argument('--femnist_train_path', type = str, default = '../data/femnist/sample_wise/train/', help = 'femnist train json dir path')
    parser.add_argument('--femnist_test_path' , type = str, default = '../data/femnist/sample_wise/test/' , help = 'femnist test json dir path' )
    parser.add_argument('--celeba_train_path' , type = str, default = '../data/celeba/sample_wise/all_data_niid_0_keep_0_train_9.json', help = 'celeba train json path')
    parser.add_argument('--celeba_test_path'  , type = str, default = '../data/celeba/sample_wise/all_data_niid_0_keep_0_test_9.json' , help = 'celeba test json path')
    parser.add_argument('--celeba_image_path' , type = str, default = '../data/celeba/img_align_celeba/', help = 'celeba image dir path')
    parser.add_argument('--shakespeare_train_path', type = str, default = '../data/shakespeare/sample_wise/all_data_niid_0_keep_0_train_9.json', help = 'shakespeare train json path')
    parser.add_argument('--shakespeare_test_path' , type = str, default = '../data/shakespeare/sample_wise/all_data_niid_0_keep_0_test_9.json' , help = 'shakespeare test json path')
    parser.add_argument('--openimage_train_path'  , type = str, default = './data/openimage/train_path.json', help = 'openimage train json path')
    parser.add_argument('--openimage_test_path'   , type = str, default = './data/openimage/test_path.json' , help = 'openimage test json path')
    parser.add_argument('--openimage_image_path'  , type = str, default = '../data/openimage/all_samples/'  , help = 'openimage image dir path')
    parser.add_argument('--cifar10_path' , type = str, default = './data/cifar10/' , help = 'cifar10 pkl dir path')
    parser.add_argument('--cifar100_path', type = str, default = './data/cifar100/', help = 'cifar100 pkl dir path')

    # special for cifar with different dirichlet parameters
    parser.add_argument('-cda', '--cifar_dirichlet_alpha', type = float, default = 0.1, help = 'alpha for dirichlet distribution for cifar dataset. -1.0 stands for iid distribution.')
    
    args = parser.parse_args()
    args.time = str(datetime.now())[5:-10]
    args.client_C = int(args.client_C) if args.client_C > 1.0 else args.client_C

    return args

def get_clients_and_server(args: argparse.Namespace) -> tuple[list[object], Server]:
    """
    Determine clients and server.

    Arguments:
        args (argparse.Namespace): parsed argument object.

    Returns:
        clients (list[Client]): list of training datasets, one dataset per client.
        server (Server): server.
    """

    match args.project:
        case 'femnist':
            args.binary = False
            if args.default:
                args.min_sample = 32
                args.client_bs = 32
                args.server_bs = 32
                args.client_lr = 3e-4
                args.server_lr = 3e-4
            
            model = CNN_femnist()
            assert(args.cut_layer <= 3)
            cut_layer = 2 if args.cut_layer <= 0 else args.cut_layer
            client_model, server_model = model_split(model, cut_layer)
            clients = get_clients_femnist(args, client_model)
                
        case 'celeba':
            args.binary = True
            if args.default:
                args.min_sample = 16
                args.client_bs = 16
                args.server_bs = 16
                args.client_lr = 1e-2
                args.server_lr = 1e-2

            model = CNN_celeba()
            assert(args.cut_layer <= 4)
            cut_layer = 2 if args.cut_layer <= 0 else args.cut_layer
            client_model, server_model = model_split(model, cut_layer)
            clients = get_clients_celeba(args, client_model)
                
        case 'shakespeare':
            args.binary = False
            if args.default:
                args.min_sample = 32
                args.client_bs = 32
                args.server_bs = 32
                args.client_lr = 3e-2
                args.server_lr = 3e-2
            
            model = LSTM_shakespeare()
            assert(args.cut_layer <= 2)
            cut_layer = 2 if args.cut_layer <= 0 else args.cut_layer
            client_model, server_model = model_split(model, cut_layer)
            clients = get_clients_shakespeare(args, client_model)

        case 'cifar100':
            args.binary = False
            if args.default:
                args.min_sample = 1
                args.client_bs = 64
                args.server_bs = 64
                args.client_lr = 1e-4
                args.server_lr = 1e-4

            # determine model 
            assert(args.resnet in [9, 18, 34, 50, 101, 152])
            if args.resnet == 9:
                model = ResNet9_cifar()
                assert(args.cut_layer <= 6)
            else:
                model = ResNet_torch_cifar(args.resnet)
                assert(args.cut_layer <= 5)
            
            cut_layer = 3 if args.cut_layer <= 0 else args.cut_layer
            client_model, server_model = model_split(model, cut_layer)
            
            # determine data distribution
            assert(args.cifar_dirichlet_alpha in [-1.0, 1.0, 0.5, 0.1, 0.01])
            if args.cifar_dirichlet_alpha == -1.0: #iid
                args.cifar_partition_path = args.cifar100_path + 'iid/partition.pkl'
            else:
                args.cifar_partition_path = args.cifar100_path + 'dirichlet_alpha_' + str(args.cifar_dirichlet_alpha) + '/partition.pkl'

            clients = get_clients_cifar(args, client_model)
            
        case _:
            raise Exception("wrong project:", args.project)
        
    # server
    server = Server(args, server_model)

    return clients, server

def weighted_avg_params(params: list[dict[str, torch.Tensor]], weights: list[int] = None) -> dict[str, torch.Tensor]:
    """
    Average params.

    Arguments:
        params (list[dict[str, torch.Tensor]]): list of params to be averaged.
        weights (list[int]): list of weights. Each weight is the number of samples of an entity.

    Returns:
        avg_state_dict (dict[str, torch.Tensor]): state_dict of averaged model.
    """
    
    if weights is None:
        weights = [1.0] * len(params)
    else:
        assert(len(params) == len(weights))
        
    avg_state_dict = copy.deepcopy(params[0])
    for key in avg_state_dict.keys():
        avg_state_dict[key] *= weights[0]
        for i in range(1, len(params)):
            avg_state_dict[key] += params[i][key] * weights[i]
        avg_state_dict[key] = torch.div(avg_state_dict[key], sum(weights))
    return avg_state_dict

def FedAvg_agg(param_entities: list[object], update_entities: list[object] = None, weights: list[int] = None) -> dict[str, torch.Tensor]:
    """
    Vanilla FL aggregation algorithm, which is often called FedAvg.

    Arguments:
        param_entities (list[Client | Server]): list of entities (client or server) which provide models.
        update_entities (list[Client | Server]): list of entities (client or server) whose models will be updated.
        weights (list[int]): list of weights. Each weight is the number of samples of an entity.
    """

    # when parameter providing entities are also updating entities
    if update_entities is None:
        update_entities = param_entities

    avg_state_dict = weighted_avg_params([e.model.state_dict() for e in param_entities], weights)
    for e in update_entities:
        e.model.load_state_dict(avg_state_dict)

def FedOpt_agg(param_entities: list[object], update_entities: list[object] = None, weights: list[int] = None) -> dict[str, torch.Tensor]:
    """
    FL aggregation algorithm FedOpt. Depending on the choice of optimizer, it can be deviated into different variates like FedAdam and FedAMS.

    Arguments:
        same as FedAvg.
    """

    # when parameter providing entities are also updating entities
    if update_entities is None:
        update_entities = param_entities

    # collect gradients and average them
    grads = []
    for e in param_entities:
        g = {}
        for p_name, p in e.model.named_parameters():
            if p.requires_grad:
                g[p_name] = p.grad
        grads.append(g)
    avg_grad = weighted_avg_params(grads, weights)
    
    # apply gradient step to one entity. Note that all param entities are update entities.
    update_entities[0].model.to(update_entities[0].device)
    for p_name, p in update_entities[0].model.named_parameters():
        if p.requires_grad:
            p.grad = avg_grad[p_name]
    update_entities[0].apply_optim()
    
    # all other entities just load state from the updated entity to avoid numerical error.
    for e in update_entities[1:]:
        e.to('cpu')
        e.model.load_state_dict(update_entities[0].model.state_dict())
        e.model.zero_grad()
        e.optim.load_state_dict(update_entities[0].optim.state_dict())