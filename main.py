import math
import torch
import wandb

# self-defined functions
from sl import *
from utils import seed, Args, get_clients_and_server

def main(args: object) -> None:
    """
    Helper function for split learning. It calls the corresponding sl algorithm in sl.py.

    Arguments:
        args (argparse.Namespace): parsed argument object. 
    """

    # reproducibility
    seed(args.seed)

    # device
    if torch.cuda.is_available() and args.device >= 0:
        args.device = 'cuda:' + str(args.device)
    else:
        args.device = 'cpu'
    print("\nusing device:", args.device)
        
    # get clients and server
    clients, server = get_clients_and_server(args)

    # some print
    # print("\nclient part model:\n", clients[0].model)
    # print("\nserver part model:\n", server.model)
    print("\nnumber of all clients with more than " + str(args.min_sample) + " samples:", len(clients))
    # print("length of train dataset:", sum([len(c.train_dataset) for c in clients]))
    # print("length of test  dataset:", sum([len(c.test_dataset ) for c in clients]))
        
    # drop some clients when the dataset contains too many of them
    num_all_client = len(clients)
    num_kept_client = min(max(math.ceil(args.keep_rate * num_all_client), 1), num_all_client)
    clients = clients[:num_kept_client]
    print("\nnumber of kept clients:", len(clients))
    
    # determine number of clients to be sampled in each round
    if args.client_C <= 1.0: # proportion
        num_sample_client = min(max(math.ceil(args.client_C * num_kept_client), 1), num_kept_client)
    else: # absolute value
        num_sample_client = min(args.client_C, num_kept_client)
    print("\nnumber of sampled clients in each round:", num_sample_client)
    print()

    # wandb init
    wandb.init(project = 'sl ' + args.project, name = args.name, config = args.__dict__, anonymous = "allow")
    
    # split learning
    print("\nrunning", args.switch_SL, ':')
    eval(args.switch_SL)(args, clients, server, num_sample_client)
    
# main function call
if __name__ == '__main__':
    args = Args()
    
    # wandb run name
    args.name  = args.switch_SL
    args.name += ': seed ' + str(args.seed)
    # args.name += '; C ' + str(args.client_C)
    # args.name += '; c_op ' + args.client_optim + ' ' + str(args.client_lr)
    # args.name += '; s_op ' + args.server_optim + ' ' + str(args.server_lr)
        
    main(args)