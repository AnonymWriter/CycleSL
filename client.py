import torch
from torch.optim import *

class Client(object):
    """
    Self-defined client class.
    """

    def __init__(self, args: object, train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, model: torch.nn.Module) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            train_dataset (torch.utils.data.Dataset): train dataset.
            test_dataset (torch.utils.data.Dataset): test dataset.
            model (torch.nn.Module): client part model.
        """

        super(Client, self).__init__()
        self.device = args.device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_sample = len(train_dataset)
        self.model = model
        self.optim = eval(args.client_optim)(self.model.parameters(), lr = args.client_lr)
        
        # data loader and iterator
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = args.client_bs, shuffle = True)
        self.test_loader  = torch.utils.data.DataLoader(self.test_dataset , batch_size = args.client_bs, shuffle = True)
        self.train_iter = iter(self.train_loader)

        # None variables at beginning
        self.x_last = None
            
    def forward_one_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pass one mini-batch samples through client local model. Iterator is used for this purpose instead of the commonly used data loader,
        in order to memorize where the iteration stopped in last round.
        
        Returns:
            x (torch.Tensor): one mini-batch features extracted by client model. x is already detached clone.
            y (torch.Tensor): one mini-batch labels.
        """

        self.to(self.device)
        self.model.train()
        
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)

        x = x.to(self.device)
        y = y.to(self.device)
        
        x = self.model(x)
        self.x_last = x
        
        return x.detach().clone(), y
    
    def backward_one_batch(self, grad: torch.Tensor) -> None:
        """
        Backward propagation using gradients sent by server.

        Arguments:
            grad (torch.Tensor): one mini-batch/sample gradient returned by server.
        """

        if self.x_last.shape == grad.shape: # batch grad
            self.x_last.backward(grad)
        else: # sample grad
            repeat_pattern = (len(self.x_last),) + (1,) * grad.dim()
            grad = grad.repeat(repeat_pattern)
            self.x_last.backward(grad)
    
    def apply_optim(self) -> None:
        """
        Apply optimizer step.
        """

        self.optim.step()
        self.optim.zero_grad()
        self.to('cpu')

    def to(self, device: str) -> None:
        """
        Move model and optimizer between CPU and GPU.

        Arguments:
            device (str): CPU or GPU.
        """
        
        # model
        self.model.to(device)

        # optimizer. Source: https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3.
        for param in self.optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)