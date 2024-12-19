import os
import json
import copy
import tqdm
import torch
import pickle
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ShuffleNet_V2_X2_0_Weights

# self-defined functions
from client import Client

class Dataset(torch.utils.data.Dataset):
    """
    Self-define dataset class. 
    """

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor, ws: int = None) -> None:
        """
        Arguments:
            xs (torch.Tensor): samples.
            ys (torch.Tensor): ground truth labels.
            ws (int | torch.Tensor): weights, which is the number of samples of a client.
        """

        assert(len(xs) == len(ys))
        self.xs = xs
        self.ys = ys
        self.ws = ws
        
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """

        return len(self.ys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx (int): index to sample.

        Returns:
            x (torch.Tensor): sample.
            y (torch.Tensor): ground truth label.
            w (torch.Tensor): weights.
        """
        if self.ws is None:
            return self.xs[idx], self.ys[idx]
        else:
            return self.xs[idx], self.ys[idx], self.ws

class Dataset_openimage(torch.utils.data.Dataset):
    """
    (deprecated) Self-define dataset class for OpenImage task. 
    """

    def __init__(self, xs: list[str], ys: torch.Tensor, image_dir: str) -> None:
        """
        Arguments:
            xs (list[str]): paths to images.
            ys (torch.Tensor): ground truth labels.
            image_dir (str): image folder path.
        """

        assert(len(xs) == len(ys))
        assert(os.path.exists(image_dir))

        self.xs = xs
        self.ys = ys
        self.t = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1.transforms()
        self.image_dir = image_dir
        
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """

        return len(self.ys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx (int): index to sample.

        Returns:
            x (torch.Tensor): sample.
            y (torch.Tensor): ground truth label.
        """
        
        x = self.xs[idx]
        x = Image.open(self.image_dir + x)
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.t(x)
        return x, self.ys[idx]

def get_clients_femnist(args: object, model: torch.nn.Module, image_size: int = 28) -> list[Client]:
    """
    Read FEMNIST data json file and get list of clients.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        model (torch.nn.Module): client part model.
        image_size (int): height / width of images. The images should be of rectangle shape.
        
    Returns:
        clients (list[Client]): a list of clients. Each client has its own train dataset and test dataset.
    """

    assert(os.path.exists(args.femnist_train_path) and os.path.exists(args.femnist_test_path))
    
    train_jsons = os.listdir(args.femnist_train_path); train_jsons.sort()
    test_jsons  = os.listdir(args.femnist_test_path ); test_jsons .sort()
    assert(len(train_jsons) == len(test_jsons))
    
    clients = []

    print('\nloading clients:')
    for train_json, test_json in tqdm.tqdm(zip(train_jsons, test_jsons)):
        with open(args.femnist_train_path + train_json, 'r') as train_f:
            train_file = json.load(train_f)
        with open(args.femnist_test_path  + test_json , 'r') as test_f :
            test_file  = json.load(test_f )
        
        for user, num_sample, user2 in zip(train_file['users'], train_file['num_samples'], test_file['users']):
            assert(user == user2)

            # discard a user if it has too few samples
            if num_sample < args.min_sample:
                continue

            # train dataset
            xs = []
            for x in train_file['user_data'][user]['x']:
                x = torch.as_tensor(x).reshape(1, image_size, image_size)
                xs.append(x)
            xs = torch.stack(xs)
            ys = torch.as_tensor(train_file['user_data'][user]['y']).long()
            train_dataset = Dataset(xs, ys)
            
            # test dataset
            xs = []
            for x in test_file['user_data'][user]['x']:
                x = torch.as_tensor(x).reshape(1, image_size, image_size)
                xs.append(x)
            xs = torch.stack(xs)
            ys = torch.as_tensor(test_file['user_data'][user]['y']).long()
            test_dataset = Dataset(xs, ys)

            client = Client(args, train_dataset, test_dataset, copy.deepcopy(model))
            clients.append(client)
        
    return clients

def get_clients_celeba(args: object, model: torch.nn.Module, image_size: int = 84) -> list[Client]:
    """
    Read CelebA images and get list of clients.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        model (torch.nn.Module): client part model.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        clients (list[Client]): a list of clients. Each client has its own train dataset and test dataset.
    """

    assert(os.path.exists(args.celeba_train_path) and os.path.exists(args.celeba_test_path) and os.path.exists(args.celeba_image_path))
    
    with open(args.celeba_train_path, 'r') as train_f:
        train_file = json.load(train_f)
    with open(args.celeba_test_path , 'r') as test_f :
        test_file  = json.load(test_f )
    
    # transformer
    t = transforms.ToTensor()

    clients = []

    print('\nloading clients:')
    for user, num_sample, user2 in tqdm.tqdm(zip(train_file['users'], train_file['num_samples'], test_file['users'])):
        assert(user == user2)

        # discard a user if it has too few samples
        if num_sample < args.min_sample:
            continue

        # train dataset
        xs = []
        for x in train_file['user_data'][user]['x']:
            x = Image.open(args.celeba_image_path + x)
            x = x.resize((image_size, image_size)).convert('RGB')
            x = t(x)
            xs.append(x)
        xs = torch.stack(xs)
        ys = torch.as_tensor(train_file['user_data'][user]['y']).long()
        train_dataset = Dataset(xs, ys)

        # test dataset
        xs = []
        for x in test_file['user_data'][user]['x']:
            x = Image.open(args.celeba_image_path + x)
            x = x.resize((image_size, image_size)).convert('RGB')
            x = t(x)
            xs.append(x)
        xs = torch.stack(xs)
        ys = torch.as_tensor(test_file['user_data'][user]['y']).long()
        test_dataset = Dataset(xs, ys)

        client = Client(args, train_dataset, test_dataset, copy.deepcopy(model))
        clients.append(client)
        
    return clients

def get_clients_shakespeare(args: object, model: torch.nn.Module, seq_length: int = 80, num_class: int = 80) -> list[Client]:
    """
    Read Shakespeare data json file and get list of clients.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        model (torch.nn.Module): client part model.
        seq_length (int): length of character sequence.
        num_class (int): number of classes (unique characters) in the dataset.

    Returns:
        clients (list[Client]): a list of clients. Each client has its own train dataset and test dataset.
    """

    assert(os.path.exists(args.shakespeare_train_path) and os.path.exists(args.shakespeare_test_path))
    
    with open(args.shakespeare_train_path, 'r') as train_f:
        train_file = json.load(train_f)
    with open(args.shakespeare_test_path , 'r') as test_f :
        test_file  = json.load(test_f )

    # all 80 chars
    all_chars_sorted = ''' !"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}'''
    # all 75 chars (not 80 chars, because some chars are missing in train/test/both datasets)
    # all_chars_sorted = ''' !"&'(),-.12345678:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz'''
    assert(len(all_chars_sorted) == num_class)
    
    clients = []
    
    print('\nloading clients:')
    for user, num_sample, user2 in tqdm.tqdm(zip(train_file['users'], train_file['num_samples'], test_file['users'])):
        assert(user == user2)

        # discard a user if it has too few samples
        if num_sample < args.min_sample:
            continue

        # train dataset
        xs, ys = [], []
        for x, y in zip(train_file['user_data'][user]['x'], train_file['user_data'][user]['y']):
            assert(len(x) == seq_length)
            y = all_chars_sorted.find(y)
            if y == -1: # cannot find character
                raise Exception('wrong character:', y)
            ys.append(y)
            seq = torch.as_tensor([all_chars_sorted.find(c) for c in x])
            xs.append(seq)
        xs = torch.stack(xs)
        ys = torch.as_tensor(ys).long()
        train_dataset = Dataset(xs, ys)

        # test dataset
        xs, ys = [], []
        for x, y in zip(test_file['user_data'][user]['x'], test_file['user_data'][user]['y']):
            assert(len(x) == seq_length)
            y = all_chars_sorted.find(y)
            if y == -1: # cannot find character
                raise Exception('wrong character:', y)
            ys.append(y)
            seq = torch.as_tensor([all_chars_sorted.find(c) for c in x])
            xs.append(seq)
        xs = torch.stack(xs)
        ys = torch.as_tensor(ys).long()
        test_dataset = Dataset(xs, ys)
        
        client = Client(args, train_dataset, test_dataset, copy.deepcopy(model))
        clients.append(client)
     
    return clients

def get_clients_cifar(args: object, model: torch.nn.Module) -> list[Client]:
    """
    Download Cifar data and get list of clients.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        model (torch.nn.Module): client part model.
        
    Returns:
        clients (list[Client]): a list of clients. Each client has its own train dataset and test dataset.
    """

    assert(os.path.exists(args.cifar_partition_path))

    print()
    if args.project == 'cifar10':
        train_samples = CIFAR10('./temp/', True , download = True)
        test_samples  = CIFAR10('./temp/', False, download = True)
    elif args.project == 'cifar100':
        train_samples = CIFAR100('./temp/', True , download = True)
        test_samples  = CIFAR100('./temp/', False, download = True)
    else:
        raise Exception("wrong project:", args.project)
    train_x = torch.Tensor(train_samples.data).permute([0, -1, 1, 2]).float()
    test_x  = torch.Tensor(test_samples .data).permute([0, -1, 1, 2]).float()
    train_y = torch.Tensor(train_samples.targets).long().squeeze()
    test_y  = torch.Tensor(test_samples .targets).long().squeeze()
    all_x = torch.cat([train_x, test_x])
    all_y = torch.cat([train_y, test_y])
    t = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.201])
    all_x = t(all_x)
    
    clients = []

    with open(args.cifar_partition_path, 'rb') as f:
        partition = pickle.load(f)
    
    print('\nloading clients:')
    for p in partition['data_indices']:
        train_idx = p['train']
        test_idx  = p['test' ]

        train_dataset = Dataset(all_x[train_idx], all_y[train_idx])
        test_dataset  = Dataset(all_x[test_idx ], all_y[test_idx ])

        client = Client(args, train_dataset, test_dataset, copy.deepcopy(model)) 
        clients.append(client)
        
    return clients