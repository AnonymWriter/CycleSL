import torch
import copy
import numpy as np
import random
import argparse
from datetime import datetime

# self-defined functions
from models import *
from data_preprocessing import get_clients_femnist, get_clients_celeba, get_clients_shakespeare
from server import Server

def seed(seed: int = 22) -> None:
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
    
    # path parameters
    parser.add_argument('--femnist_train_path', type = str, default = '/data/femnist/train/', help = 'femnist train json dir path')
    parser.add_argument('--femnist_test_path' , type = str, default = '/data/femnist/test/' , help = 'femnist test json dir path' )
    parser.add_argument('--celeba_train_path' , type = str, default = '/data/celeba/train/all_data_niid_0_keep_0_train_9.json', help = 'celeba train json path')
    parser.add_argument('--celeba_test_path'  , type = str, default = '/data/celeba/test/all_data_niid_0_keep_0_test_9.json'  , help = 'celeba test json path')
    parser.add_argument('--celeba_image_path' , type = str, default = '/data/celeba/img_align_celeba/', help = 'celeba image dir path')
    parser.add_argument('--shakespeare_train_path', type = str, default = '/data/shakespeare/train/all_data_niid_0_keep_0_train_9.json', help = 'shakespeare train json path')
    parser.add_argument('--shakespeare_test_path' , type = str, default = '/data/shakespeare/test/all_data_niid_0_keep_0_test_9.json'  , help = 'shakespeare test  json path')
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'femnist', help = 'project name, from femnist, celeba, shakespeare, covid19')
    parser.add_argument('-seed', '--seed', type = int, default = 22, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-SL', '--switch_SL', type = str, default = 'SSL', help = 'SL algorithm')
    
    # hyperparameters
    parser.add_argument('-D', '--default', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use default hyperparmeters')
    parser.add_argument('-T', '--num_round', type = int, default = 601, help = 'number of split learning rounds')
    parser.add_argument('-E', '--num_server_epoch', type = int, default = 1, help = 'number of server epochs')
    parser.add_argument('-K', '--keep_rate', type = float, default = 1.0, help = 'proportion of clients considered. Useful when the dataset contains too many clients.')
    parser.add_argument('-C', '--client_C', type = float, default = 0.05, help = 'number / proportion of participating clients in each aggregation round')
    parser.add_argument('-S', '--min_sample', type = int, default = 32, help = 'minimal required amount of samples per client')
    parser.add_argument('-c_bs', '--client_bs', type = int, default = 32, help = 'batch size for client training data loader')
    parser.add_argument('-s_bs', '--server_bs', type = int, default = 32, help = 'batch size for server training data loader')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = 3e-4, help = 'client learning rate')
    parser.add_argument('-s_lr', '--server_lr', type = float, default = 3e-4, help = 'server learning rate')
    parser.add_argument('-c_op', '--client_optim', type = str, default = 'Adam', help = 'client optimizer')
    parser.add_argument('-s_op', '--server_optim', type = str, default = 'Adam', help = 'server optimizer')
    
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
            client_model = CNN_femnist_client()
            server_model = CNN_femnist_server()
            if args.default:
                args.min_sample = 32
                args.client_bs = 32
                args.server_bs = 32
                args.client_lr = 3e-4
                args.server_lr = 3e-4
            clients = get_clients_femnist(args, client_model)
                
        case 'celeba':
            args.binary = True
            client_model = CNN_celeba_client()
            server_model = CNN_celeba_server()
            if args.default:
                args.min_sample = 16
                args.client_bs = 16
                args.server_bs = 16
                args.client_lr = 1e-2
                args.server_lr = 1e-2
            clients = get_clients_celeba(args, client_model)
                
        case 'shakespeare':
            args.binary = False
            client_model = LSTM_shakespeare_client()
            server_model = LSTM_shakespeare_server()
            if args.default:
                args.min_sample = 64
                args.client_bs = 64
                args.server_bs = 64
                args.client_lr = 1e-2
                args.server_lr = 1e-2
            clients = get_clients_shakespeare(args, client_model)
                
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