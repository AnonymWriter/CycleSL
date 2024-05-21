import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score
import wandb

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'

# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef}

class CNN_femnist_client(torch.nn.Module):
    """
    Client part model for FEMNIST task. The model structure follows the LEAF framework.
    """

    def __init__(self, image_size: int = 28, num_class: int = 62) -> None:
        """
        Arguments:
            image_size (int): height / width of images. The images should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(CNN_femnist_client, self).__init__()

        self.seq = torch.nn.Sequential(
            # client part
            torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            # server part
            # torch.nn.Flatten(),
            # torch.nn.Linear(in_features = 64 * int(image_size / 4) * int(image_size / 4), out_features = 2048),
            # torch.nn.ReLU(),

            # torch.nn.Linear(in_features = 2048, out_features = num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): features.
        """
        
        x = self.seq(x)
        return x
    
class CNN_femnist_server(torch.nn.Module):
    """
    Server part model for FEMNIST task. The model structure follows the LEAF framework.
    """

    def __init__(self, image_size: int = 28, num_class: int = 62) -> None:
        """
        Arguments:
            image_size (int): height / width of images. The images should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(CNN_femnist_server, self).__init__()

        self.seq = torch.nn.Sequential(
            # client part
            # torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, padding = 'same'),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            # torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same'),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            # server part
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 64 * int(image_size / 4) * int(image_size / 4), out_features = 2048),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features = 2048, out_features = num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): features.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
        """
        
        x = self.seq(x)
        return x
    
class CNN_celeba_client(torch.nn.Module):
    """
    Client part model for CelebA task. The model structure follows the LEAF framework.
    """

    def __init__(self, image_size: int = 84, num_class: int = 2) -> None:
        """
        Arguments:
            image_size (int): height / width of image. The image should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(CNN_celeba_client, self).__init__()

        self.seq = torch.nn.Sequential(
            # client part
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            # server part
            # torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            # torch.nn.BatchNorm2d(num_features = 32),
            # torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            # torch.nn.ReLU(),

            # torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            # torch.nn.BatchNorm2d(num_features = 32),
            # torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            # torch.nn.ReLU(),

            # torch.nn.Flatten(),
            # torch.nn.Linear(in_features = 32 * int(image_size / 16) * int(image_size / 16), out_features = num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): features.
        """

        x = self.seq(x)
        return x
    
class CNN_celeba_server(torch.nn.Module):
    """
    Server part model for CelebA task. The model structure follows the LEAF framework.
    """

    def __init__(self, image_size: int = 84, num_class: int = 2) -> None:
        """
        Arguments:
            image_size (int): height / width of image. The image should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(CNN_celeba_server, self).__init__()

        self.seq = torch.nn.Sequential(
            # client part
            # torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 'same'),
            # torch.nn.BatchNorm2d(num_features = 32),
            # torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            # torch.nn.ReLU(),

            # torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            # torch.nn.BatchNorm2d(num_features = 32),
            # torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            # torch.nn.ReLU(),

            # server part
            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 32 * int(image_size / 16) * int(image_size / 16), out_features = num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): features.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
        """

        x = self.seq(x)
        return x
    
class LSTM_shakespeare_client(torch.nn.Module):
    """
    Client part model for Shakespeare dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, embedding_dim: int = 8, hidden_size: int = 256, num_class: int = 80) -> None:
        """
        Arguments:
            embedding_dim (int): dimension of character embedding.
            hidden_size (int): dimension of LSTM hidden state.
            num_class (int): number of classes (unique characters) in the dataset.
        """

        super(LSTM_shakespeare_client, self).__init__()

        # client part
        self.embedding = torch.nn.Embedding(num_embeddings = num_class, embedding_dim = embedding_dim)
        self.encoder = torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, batch_first = True)
        
        # server part
        # self.logits = torch.nn.Linear(in_features = hidden_size, out_features = num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): index to embeddings.
        
        Returns:
            h (torch.Tensor): features.
        """

        x = self.embedding(x)
        x, (hn, cn) = self.encoder(x)
        h = x[:, -1, :]
        return h
    
class LSTM_shakespeare_server(torch.nn.Module):
    """
    Server part model for Shakespeare dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, embedding_dim: int = 8, hidden_size: int = 256, num_class: int = 80) -> None:
        """
        Arguments:
            embedding_dim (int): dimension of character embedding.
            hidden_size (int): dimension of LSTM hidden state.
            num_class (int): number of classes (unique characters) in the dataset.
        """

        super(LSTM_shakespeare_server, self).__init__()

        # client part
        # self.embedding = torch.nn.Embedding(num_embeddings = num_class, embedding_dim = embedding_dim)
        # self.encoder = torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, batch_first = True)
        
        # server part
        self.logits = torch.nn.Linear(in_features = hidden_size, out_features = num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): features.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
        """
        
        x = self.logits(x)
        return x

def model_eval(args: object,
               clients: list[object], 
               server: object,
               use_train_dataset: bool,
               wandb_log: dict[str, float], 
               metric_prefix: str = 'prefix/',
               ) -> None:
    """
    Evaludate the performance of a full model with differnt metrics (loss, accuracy, MCC score, precision, recall, F1 score).

    Arguments:
        args (argparse.Namespace): parsed argument object.
        clients (list[Client]): list of clients.
        server (Server): server.
        use_train_dataset (bool): whether to use train loader or test loader.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """

    round_labels = []
    round_preds  = []

    server.model.to(args.device)
    server.model.eval()

    with torch.no_grad():
        for c in clients:
            c.model.to(args.device)
            c.model.eval()
    
            loader = c.train_loader if use_train_dataset else c.test_loader
            for batch_id, (x, y) in enumerate(loader):
                x = x.to(args.device)
                y = y.to(args.device)
                
                h = c.model(x)
                preds = server.model(h)
                
                round_labels.append(y)
                round_preds.append(preds)
            
            c.model.to('cpu')
        
    server.model.to('cpu')

    round_labels = torch.cat(round_labels).detach().to('cpu')
    round_preds  = torch.cat(round_preds ).detach().to('cpu')

    cal_metrics(round_labels, round_preds, args.binary, wandb_log, metric_prefix)

def cal_metrics(labels: torch.Tensor, preds: torch.Tensor, binary: bool, wandb_log: dict[str, float], metric_prefix: str) -> None:
    """
    Compute metrics (loss, accuracy, MCC score, precision, recall, F1 score) using ground truth labels and logits.

    Arguments:
        labels (torch.Tensor): ground truth labels.
        preds (torch.Tensor): logits (not softmaxed yet).
        binary (bool): whether doing binary classification or multi-class classification.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """

    # loss
    loss = F.cross_entropy(preds, labels)
    wandb_log[metric_prefix + 'loss'] = loss
        
    if not binary: # multi-class    
        # get probability
        preds = torch.softmax(preds, axis = 1)

        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds, multi_class = 'ovr')
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1

        # get class prediction
        preds = preds.argmax(axis = 1)
        
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric
    
    else: # binary
        # get probability
        preds = torch.softmax(preds, axis = 1)[:, 1]
        
        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds)
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1
        
        # get class prediction
        preds = preds.round()
        
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric

def eval_help(args: object, clients: list[object], server: object) -> None:
    """
    Evaluatition helper.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        clients (list[Client]): list of clients.
        server (Server): server.
    """
    
    wandb_log = {}
    model_eval(args, clients, server, True , wandb_log, 'train/')
    model_eval(args, clients, server, False, wandb_log, 'test/' )
    wandb.log(wandb_log)