# set seed for everything
# from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# from https://github.com/pytorch/examples/blob/main/imagenet/main.py#L199
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# minimal example
def loop(data_loader, model, criterion, args):
    a_loss = AverageMeter() # instantiate
    for _, batch in enumerate(data_loader):
        x, y = batch
        logits = model(x)
        loss = criterion(logits, y)
        a_loss.update(loss.item()) # update with value
        
    avg_val_loss = a_loss.avg # retrieve average value
    return avg_val_loss