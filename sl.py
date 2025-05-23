import tqdm
import copy
import torch
import numpy as np
from torch.optim import *
import torch.nn.functional as F

# self-defined functions
from models import eval_help
from utils import FedAvg_agg, FedOpt_agg
from data_preprocessing import Dataset

def SSL(args: object, clients: list[object], server: object, num_sample_client: int) -> None:
    """
    Main loop for vanilla split learning, which is also called sequential split learning.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        clients (list[Client]): list of clients.
        server (Server): server.
        num_sample_client (int): how many clients are involved in one round.
    """

    # in SSL, every client trains the latest model
    latest_client_model = copy.deepcopy(clients[0].model)
    
    # evaluate model performance before training
    eval_help(args, clients, server)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        for c in sample_clients:
            # client side
            c.model.load_state_dict(latest_client_model.state_dict())
            x, y = c.forward_one_batch()
            
            # server side
            grad = server.grad_one_batch(x, y)
            server.apply_optim()
            
            # client side
            c.backward_one_batch(grad)
            c.apply_optim()
            latest_client_model.load_state_dict(c.model.state_dict())
        
        # evaluation
        eval_help(args, sample_clients, server)

def PSL(args: object, clients: list[object], server: object, num_sample_client: int) -> None:
    """
    Main loop for parallel split learning.

    Arguments:
        same as SSL.
    """

    # in PSL, server has multiple copies of server part model. We emulate this by having multiple servers.
    servers = [copy.deepcopy(server) for i in range(num_sample_client)]

    # evaluate model performance before training
    eval_help(args, clients, servers[0])
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        for c, s in zip(sample_clients, servers):
            # client side
            x, y = c.forward_one_batch()
            
            # server side
            grad = s.grad_one_batch(x, y)
            s.apply_optim()
            
            # client side
            c.backward_one_batch(grad)
            c.apply_optim()
        
        # aggregate server models
        FedAvg_agg(servers, weights = [c.num_sample for c in sample_clients])
                    
        # evaluation. For a fair comparison of model generalization, we use local model of a random client as client part.
        # note that all server models are identical
        eval_help(args, sample_clients, servers[0])

def SFLV1(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    """
    Main loop for split federated learning (SplitFed) version 2.

    Arguments:
        same as SSL.
    """

    # in SFLV1, server has multiple copies of server part model. We emulate this by having multiple servers.
    servers = [copy.deepcopy(server) for i in range(num_sample_client)]

    # evaluate model performance before training
    eval_help(args, clients, servers[0])
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        for c, s in zip(sample_clients, servers):
            # client side
            x, y = c.forward_one_batch()
            
            # server side
            grad = s.grad_one_batch(x, y)
            s.apply_optim()
            
            # client side
            c.backward_one_batch(grad)
            c.apply_optim()
            
        # aggregate server models
        FedAvg_agg(servers, weights = [c.num_sample for c in sample_clients])

        # aggregate client models
        if agg_all_clients: # aggregation influences all clients (also those clients who are not sampled in this round)
            FedAvg_agg(sample_clients, clients, weights = [c.num_sample for c in sample_clients])
        else: # aggregation influences only sampled clients
            FedAvg_agg(sample_clients, weights = [c.num_sample for c in sample_clients])
            
        # evaluation. Note that all client models are identical and all server models are identical.
        eval_help(args, sample_clients, servers[0])

def SFLV2(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    """
    Main loop for split federated learning (SplitFed) version 2.

    Arguments:
        same as SSL.
    """

    # evaluate model performance before training
    eval_help(args, clients, server)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        for c in sample_clients:
            # client side
            x, y = c.forward_one_batch()
            
            # server side
            grad = server.grad_one_batch(x, y)
            server.apply_optim()
            
            # client side
            c.backward_one_batch(grad)
            c.apply_optim()
            
        # aggregate client models
        if agg_all_clients: # aggregation influences all clients (also those clients who are not sampled in this round)
            FedAvg_agg(sample_clients, clients, weights = [c.num_sample for c in sample_clients])
        else: # aggregation influences only sampled clients
            FedAvg_agg(sample_clients, weights = [c.num_sample for c in sample_clients])
            
        # evaluation. Note that all client models are identical and all server model are identical.
        eval_help(args, sample_clients, server)

def SGLR(args: object, clients: list[object], server: object, num_sample_client: int, active_C: float = 0.5) -> None:
    """
    Main loop for server-Side local gradient averaging and learning rate acceleration (SGLR, which combines SLR and SGL). 

    Arguments:
        same as SSL.
    """

    # in SGLR, server has multiple copies of server part model. We emulate this by having multiple servers.
    servers = [copy.deepcopy(server) for i in range(num_sample_client)]

    # evaluate model performance before training
    eval_help(args, clients, servers[0])

    # active average clients in SGLR
    num_active_client = int(num_sample_client * active_C)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        grads = []
        weights = []
        i = 0
        for c, s in zip(sample_clients, servers):
            # client side
            x, y = c.forward_one_batch()
            
            # server side
            grad = s.grad_one_batch(x, y)
            s.apply_optim()

            if i < num_active_client: # active clients
                c.to('cpu')
                grads.append(grad)
                weights.append(c.num_sample)
            else: # inactive clients
                c.backward_one_batch(grad)
                c.apply_optim()
            i += 1

        # aggregate server models
        FedAvg_agg(servers, weights = [c.num_sample for c in sample_clients])

        # average gradients
        sum_grad = 0
        sum_w = 0
        for grad, w in zip(grads, weights):
            # unroll batch
            for sample_grad in grad:
                sum_grad += sample_grad * w
                sum_w += w
        avg_grad = sum_grad / sum_w

        # update active clients
        for c in sample_clients[:num_active_client]:
            c.to(args.device)
            c.backward_one_batch(avg_grad)
            c.apply_optim()
                        
        # evaluation. Note all server models are identical.
        eval_help(args, sample_clients, servers[0])

def FedAvg(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    """
    Main loop for vanilla federated learning (FedAvg).

    Arguments:
        same as SSL.
    """

    # we emulate FL (FedAvg) with client-server pairs. A pair has a full model.
    servers = [copy.deepcopy(server) for c in clients]
    
    # evaluate model performance before training
    eval_help(args, clients, servers[0])
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample pairs which are involved in this round
        sample_idx = np.random.choice(len(clients), num_sample_client, replace = False)
        sample_clients = [clients[i] for i in sample_idx]
        sample_servers = [servers[i] for i in sample_idx]

        for c, s in zip(sample_clients, sample_servers):
            x, y = c.forward_one_batch()
            grad = s.grad_one_batch(x, y)
            s.apply_optim()
            c.backward_one_batch(grad)
            c.apply_optim()

        # aggregation
        if agg_all_clients: # aggregation influences all clients (also those clients who are not sampled in this round)
            FedAvg_agg(sample_clients, clients, weights = [c.num_sample for c in sample_clients])
            FedAvg_agg(sample_servers, servers, weights = [c.num_sample for c in sample_clients])
        else: # aggregation influences only sampled clients
            FedAvg_agg(sample_clients, weights = [c.num_sample for c in sample_clients])
            FedAvg_agg(sample_servers, weights = [c.num_sample for c in sample_clients])
        
        # evaluation
        eval_help(args, sample_clients, sample_servers[0])

def CyclePSL(args: object, clients: list[object], server: object, num_sample_client: int) -> None:
    """
    Main loop for our method + PSL.

    Arguments:
        same as SSL.
    """

    # evaluate model performance before training
    eval_help(args, clients, server)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        # feature datasets
        feature_datasets = []
        for c in sample_clients:
            # client side
            x, y = c.forward_one_batch()
            x = x.to('cpu')
            y = y.to('cpu')
            c.to('cpu')
            
            # collect features, labels, and weights from clients, and form datasets
            d = Dataset(x, y, c.num_sample)
            feature_datasets.append(d)
        
        # form dataset and dataloader for 2nd level machine learning (expectation maximization)
        server_dataset = torch.utils.data.ConcatDataset(feature_datasets)
        server_loader  = torch.utils.data.DataLoader(server_dataset, batch_size = args.server_bs, shuffle = True)

        # 2nd level ML on server side
        server.to(args.device)
        server.freeze(False)
        server.model.train()
        for current_server_epoch in range(args.num_server_epoch):
            for batch_id, (x, y, w) in enumerate(server_loader):
                x = x.to(args.device)
                y = y.to(args.device)
                w = w.to(args.device)

                server.optim.zero_grad()
                preds = server.model(x)
                
                # weight loss
                loss = F.cross_entropy(preds, y, reduction = 'none')
                loss = sum(loss * w) / sum(w)
                
                loss.backward()
                server.optim.step()
            
        # freeze server model and re-use client features to optimize client models
        server.model.zero_grad()
        server.freeze(True)
        for c, d in zip(sample_clients, feature_datasets):
            grad = server.grad_one_batch(d.xs.to(args.device), d.ys.to(args.device))
            c.to(args.device)
            c.backward_one_batch(grad)
            c.apply_optim()

        # evaluation
        eval_help(args, sample_clients, server)

def CycleSFL(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    """
    Main loop for our method + SFL / FSL.

    Arguments:
        same as SSL.
    """

    # evaluate model performance before training
    eval_help(args, clients, server)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        # feature datasets
        feature_datasets = []
        for c in sample_clients:
            # client side
            x, y = c.forward_one_batch()
            x = x.to('cpu')
            y = y.to('cpu')
            c.to('cpu')
            
            # collect features, labels, and weights from clients, and form datasets
            d = Dataset(x, y, c.num_sample)
            feature_datasets.append(d)
        
        # form dataset and dataloader for 2nd level machine learning (expectation maximization)
        server_dataset = torch.utils.data.ConcatDataset(feature_datasets)
        server_loader  = torch.utils.data.DataLoader(server_dataset, batch_size = args.server_bs, shuffle = True)

        # higher level ML on server side
        server.to(args.device)
        server.freeze(False)
        server.model.train()
        for current_server_epoch in range(args.num_server_epoch):
            for batch_id, (x, y, w) in enumerate(server_loader):
                x = x.to(args.device)
                y = y.to(args.device)
                w = w.to(args.device)

                server.optim.zero_grad()
                preds = server.model(x)
                
                # weight loss
                loss = F.cross_entropy(preds, y, reduction = 'none')
                loss = sum(loss * w) / sum(w)
                
                loss.backward()
                server.optim.step()
            
        # freeze server model and re-use client features to optimize client models
        server.model.zero_grad()
        server.freeze(True)
        for c, d in zip(sample_clients, feature_datasets):
            grad = server.grad_one_batch(d.xs.to(args.device), d.ys.to(args.device))
            c.to(args.device)
            c.backward_one_batch(grad)
            c.apply_optim()
        
        # aggregate client models
        if agg_all_clients: # aggregation influences also not sampled clients
            FedAvg_agg(sample_clients, clients, weights = [c.num_sample for c in sample_clients])
        else: # aggregation influences only sampled clients
            FedAvg_agg(sample_clients, weights = [c.num_sample for c in sample_clients])
        
        # evaluation
        eval_help(args, sample_clients, server)

def CycleSGLR(args: object, clients: list[object], server: object, num_sample_client: int, active_C: float = 0.5) -> None:
    """
    Main loop for our method + SGLR.

    Arguments:
        same as SSL.
    """

    # evaluate model performance before training
    eval_help(args, clients, server)

    # active average clients in SGLR
    num_active_client = int(num_sample_client * active_C)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        # feature datasets
        feature_datasets = []
        for c in sample_clients:
            # client side
            x, y = c.forward_one_batch()
            x = x.to('cpu')
            y = y.to('cpu')
            c.to('cpu')
            
            # collect features, labels, and weights from clients, and form datasets
            d = Dataset(x, y, c.num_sample)
            feature_datasets.append(d)
        
        # form dataset and dataloader for 2nd level machine learning (expectation maximization)
        server_dataset = torch.utils.data.ConcatDataset(feature_datasets)
        server_loader  = torch.utils.data.DataLoader(server_dataset, batch_size = args.server_bs, shuffle = True)

        # 2nd level ML on server side
        server.to(args.device)
        server.freeze(False)
        server.model.train()
        for current_server_epoch in range(args.num_server_epoch):
            for batch_id, (x, y, w) in enumerate(server_loader):
                x = x.to(args.device)
                y = y.to(args.device)
                w = w.to(args.device)

                server.optim.zero_grad()
                preds = server.model(x)
                
                # weight loss
                loss = F.cross_entropy(preds, y, reduction = 'none')
                loss = sum(loss * w) / sum(w)
                
                loss.backward()
                server.optim.step()
                
        # freeze server model and re-use client features to optimize client models
        server.model.zero_grad()
        server.freeze(True)
        grads = []
        weights = []
        i = 0
        for c, d in zip(sample_clients, feature_datasets):
            grad = server.grad_one_batch(d.xs.to(args.device), d.ys.to(args.device))
            
            if i < num_active_client: # active clients
                grads.append(grad.to('cpu'))
                weights.append(c.num_sample)
            else: # inactive clients
                c.to(args.device)
                c.backward_one_batch(grad)
                c.apply_optim()
            i += 1

        # average gradients
        sum_grad = 0
        sum_w = 0
        for grad, w in zip(grads, weights):
            # unroll batch
            for sample_grad in grad:
                sum_grad += sample_grad * w
                sum_w += w
        avg_grad = sum_grad / sum_w
        avg_grad = avg_grad.to(args.device)

        # update active clients
        for c in sample_clients[:num_active_client]:
            c.to(args.device)
            c.backward_one_batch(avg_grad)
            c.apply_optim()

        # evaluation
        eval_help(args, sample_clients, server)

def CycleSSL(args: object, clients: list[object], server: object, num_sample_client: int) -> None:
    """
    Main loop for our method + SSL.

    Arguments:
        same as SSL.
    """

    # in SSL, every client trains the latest model
    latest_client_model = copy.deepcopy(clients[0].model)

    # evaluate model performance before training
    eval_help(args, clients, server)
    
    # training loop
    print()
    for current_round in tqdm.tqdm(range(args.num_round)):
        # sample clients which are involved in this round
        sample_clients = np.random.choice(clients, num_sample_client, replace = False)
        
        for c in sample_clients:
            # client side
            c.model.load_state_dict(latest_client_model.state_dict())
            x, y = c.forward_one_batch()
            c.to('cpu')
            
            # server side
            server.to(args.device)
            server.freeze(False)
            server.model.train()
            for current_server_epoch in range(args.num_server_epoch):
                server.optim.zero_grad()
                preds = server.model(x)
                loss = F.cross_entropy(preds, y)
                loss.backward()
                server.optim.step()
            
            server.model.zero_grad()
            server.freeze(True)
            grad = server.grad_one_batch(x, y)
            
            # client side
            c.to(args.device)
            c.backward_one_batch(grad)
            c.apply_optim()
            latest_client_model.load_state_dict(c.model.state_dict())
        
        # evaluation
        eval_help(args, sample_clients, server)

######################################## alias functions ########################################
def SL(args: object, clients: list[object], server: object, num_sample_client: int) -> None:
    SSL(args, clients, server, num_sample_client)

def FSL(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    SFLV1(args, clients, server, num_sample_client, agg_all_clients)

def SplitFedV1(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    SFLV1(args, clients, server, num_sample_client, agg_all_clients)

def SplitFedV2(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    SFLV2(args, clients, server, num_sample_client, agg_all_clients)

def FL(args: object, clients: list[object], server: object, num_sample_client: int, agg_all_clients: bool = True) -> None:
    FedAvg(args, clients, server, num_sample_client, agg_all_clients)

def CycleFSL(args: object, clients: list[object], server: object, num_sample_client: int) -> None:
    CycleSFL(args, clients, server, num_sample_client)