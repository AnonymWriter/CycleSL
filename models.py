import wandb
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'

# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef}

class CNN_femnist(torch.nn.Module):
    """
    CNN model for FEMNIST task. The model structure follows the LEAF framework.
    """

    def __init__(self, image_size: int = 28, num_class: int = 62) -> None:
        """
        Arguments:
            image_size (int): height / width of images. The images should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super().__init__()

        self.seq = torch.nn.Sequential(
            # conv1
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, padding = 'same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
            ),
            
            # conv2
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
            ),
            
            # linear1
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(in_features = 64 * int(image_size / 4) * int(image_size / 4), out_features = 2048),
                torch.nn.ReLU()
            ),
            
            # linear2
            torch.nn.Linear(in_features = 2048, out_features = num_class)
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
    
class CNN_celeba(torch.nn.Module):
    """
    CNN model for CelebA task. The model structure follows the LEAF framework.
    """

    def __init__(self, image_size: int = 84, num_class: int = 2) -> None:
        """
        Arguments:
            image_size (int): height / width of image. The image should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super().__init__()

        self.seq = torch.nn.Sequential(
            # conv1
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 'same'),
                torch.nn.BatchNorm2d(num_features = 32),
                torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
                torch.nn.ReLU()
            ),

            # conv2
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
                torch.nn.BatchNorm2d(num_features = 32),
                torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
                torch.nn.ReLU(),
            ),

            # conv3
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
                torch.nn.BatchNorm2d(num_features = 32),
                torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
                torch.nn.ReLU()
            ),

            # conv4
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
                torch.nn.BatchNorm2d(num_features = 32),
                torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
                torch.nn.ReLU()
            ),

            # head
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(in_features = 32 * int(image_size / 16) * int(image_size / 16), out_features = num_class)
            )
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
    
class LSTM_proxy(torch.nn.Module):
    """
    A proxy layer for LSTM. This class is used to achieve a simple feed-forward LSTM which can be cut easily at any layer.
    """

    def __init__(self, lstm: torch.nn.Module) -> None:
        """
        Arguments:
            lstm (torch.nn.Module): LSTM model.
        """

        super().__init__()
        self.lstm = lstm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): embeddings.

        Returns:
            h (torch.Tensor): features.
        """

        x, (hn, cn) = self.lstm(x)
        h = x[:, -1, :]
        return h
    
class LSTM_shakespeare(torch.nn.Module):
    """
    LSTM model for Shakespeare dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, embedding_dim: int = 8, hidden_size: int = 256, num_class: int = 80) -> None:
        """
        Arguments:
            embedding_dim (int): dimension of character embedding.
            hidden_size (int): dimension of LSTM hidden state.
            num_class (int): number of classes (unique characters) in the dataset.
        """

        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings = num_class, embedding_dim = embedding_dim), 
            LSTM_proxy(torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, batch_first = True)),
            torch.nn.Linear(in_features = hidden_size, out_features = num_class)
        )
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): index to embeddings.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
        """

        x = self.seq(x)
        return x
    
def conv_block_resnet9(in_channels: int, out_channels: int, pooling: bool = False) -> torch.nn.Module:
    """
    Conv block for ResNet. Source: https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min?scriptVersionId=38462746&cellId=28.

    Arguments:
        in_channels (int): input channels.
        out_channels (int): output channels.
        pooling (bool): whether applying max pooling.

    Returns:
        seq (torch.nn.Module): conv block.
    """

    seq = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), 
        torch.nn.BatchNorm2d(out_channels), 
        torch.nn.ReLU(inplace = True)
    )
    
    if pooling:
        seq = seq.append(torch.nn.MaxPool2d(2))
    
    return seq

class Skip_proxy(torch.nn.Module):
    """
    A skip layer proxy for residual blocks.
    """

    def __init__(self, block: torch.nn.Module) -> None:
        """
        Arguments:
        block (torch.nn.Module): residual block.
        """
        
        super().__init__()
        self.block = block
    
    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x (torch.Tensor): input tensor.

        Returns:
            x (torch.Tensor): output tensor with skip connection.
        """

        x = self.block(x) + x
        return x

class ResNet9_cifar(torch.nn.Module):
    """
    ResNet9 model for Cifar task. Source: https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min?scriptVersionId=38462746&cellId=28.
    """

    def __init__(self, in_channels: int = 3, num_class: int = 100):
        """
        Arguments:
            in_channels (int): input image channels.
            num_class (int): number of classes in the dataset.
        """

        super().__init__()
        
        self.seq = torch.nn.Sequential(
            # conv1
            conv_block_resnet9(in_channels, 64),
            
            # conv2
            conv_block_resnet9(64, 128, True),
            
            # res1
            Skip_proxy(
                torch.nn.Sequential(
                    conv_block_resnet9(128, 128), 
                    conv_block_resnet9(128, 128)
                )
            ),
            
            # conv3
            conv_block_resnet9(128, 256, True),
            
            # conv4
            conv_block_resnet9(256, 512, True),
            
            # res2
            Skip_proxy(
                torch.nn.Sequential(
                    conv_block_resnet9(512, 512), 
                    conv_block_resnet9(512, 512)
                )
            ),
        
            # head
            torch.nn.Sequential(
                torch.nn.MaxPool2d(4), 
                torch.nn.Flatten(), 
                torch.nn.Linear(512, num_class)
            )
        )
        
    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
        """

        x = self.seq(x)
        return x
    
class ResNet_torch_cifar(torch.nn.Module):
    """
    A proxy for the torchvision ResNet models for Cifar task.
    """

    def __init__(self, resnet: int, num_class: int = 100) -> None:
        """
        Arguments: 
            resnet (int): which version of resnet will be used. Should be 18, 34, 50, 101, 152.
            num_class (int): number of classes in the dataset.
        """

        super().__init__()

        # determine projection head input dimension
        if resnet in [18, 34]:
            fc_feature = 512
        elif resnet in [50, 101, 152]: 
            fc_feature = 2048
        else:
            raise Exception('wrong resnet version:', resnet)
        
        m = eval('resnet' + str(resnet))(weights = None)

        self.seq = torch.nn.Sequential(
            # conv1
            torch.nn.Sequential(
                m.conv1,
                m.bn1,
                m.relu,
                m.maxpool
            ),

            # res1
            m.layer1,

            # res2
            m.layer2,

            # res3
            m.layer3,

            # res4
            m.layer4,

            # head
            torch.nn.Sequential(
                m.avgpool,
                torch.nn.Flatten(),
                torch.nn.Linear(fc_feature, num_class)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
        """

        x = self.seq(x)
        return x        
    
def model_split(model: torch.nn.Module, cut_layer: int):
    """
    Cut a model at a special layer.
    
    Arguments:
        model (torch.nn.Module): pytorch model with sequential main body, i.e., model.seq.
        cut_layer (int): cut point.

    Returns:
        client_model (torch.nn.Module): first half of the model.
        server_model (torch.nn.Module): second half of the model.
    """

    client_model = model.seq[:cut_layer]
    server_model = model.seq[cut_layer:]
    return client_model, server_model

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
                # y = y.to(args.device)
                
                h = c.model(x)
                preds = server.model(h)
                
                round_labels.append(y)
                round_preds.append(preds.to('cpu'))
            
            c.model.to('cpu')
        
    server.model.to('cpu')

    round_labels = torch.cat(round_labels).detach()
    round_preds  = torch.cat(round_preds ).detach()

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
    if not args.fast_eval:
        model_eval(args, clients, server, True , wandb_log, 'train/') # tips: skip evaluation for training samples to save time. 
    model_eval(args, clients, server, False, wandb_log, 'test/' )
    wandb.log(wandb_log)
