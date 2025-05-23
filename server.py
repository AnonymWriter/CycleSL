import torch
from torch.optim import *
import torch.nn.functional as F

class Server(object):
    """
    Self-defined server class.
    """

    def __init__(self, args: object, model: torch.nn.Module) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            model (torch.nn.Module): server part model.
            optim (str): optimizer.
        """

        super(Server, self).__init__()
        self.device = args.device
        self.model = model
        self.optim = eval(args.server_optim)(self.model.parameters(), lr = args.server_lr)
            
    def grad_one_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Pass one mini-batch features through server model and compute gradient, but not apply gradient step. This
        function should be called before apply_optim.
        
        Arguments: 
            x (torch.Tensor): one mini-batch features extracted by client model. x should be detached clone.
            y (torch.Tensor): one mini-batch labels.

        Returns:
            grad (torch.Tensor): one mini-batch gradients that are detached clone.
        """

        self.to(self.device)
        self.model.train()
        
        x.requires_grad_(True)
        preds = self.model(x)
        loss = F.cross_entropy(preds, y)
        loss.backward()
        grad = x.grad.detach().clone()
        
        return grad
    
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

    def freeze(self, freeze: bool) -> None:
        """
        freeze / unfreeze model.

        Arguments:
            freeze (bool): freeze or unfreeze.
        """

        for param in self.model.parameters():
            param.requires_grad = not freeze