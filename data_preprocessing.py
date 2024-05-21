import os
import json
import torch
import copy
from PIL import Image
import torchvision.transforms as transforms

# self-defined functions
from client import Client

class Dataset(torch.utils.data.Dataset):
    """
    Self-defined dataset class.
    """

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        """
        Arguments:
            xs (torch.Tensor): samples.
            ys (torch.Tensor): ground truth labels.
        """

        assert(len(xs) == len(ys)) 
        self.xs = xs
        self.ys = ys
        
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

        return self.xs[idx], self.ys[idx]

class Feature_Dataset(torch.utils.data.Dataset):
    """
    Self-define dataset with sample-wise weights. Useful for our SL methods which model the training
    on the server side as a standalone higher level machine learning task.
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

    for train_json, test_json in zip(train_jsons, test_jsons):
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

    for user, num_sample, user2 in zip(train_file['users'], train_file['num_samples'], test_file['users']):
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
    
    for user, num_sample, user2 in zip(train_file['users'], train_file['num_samples'], test_file['users']):
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