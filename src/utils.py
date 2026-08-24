import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter:
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
        self.avg = self.sum / (self.count + 1e-12)

def fix_batch_shapes(batch):
    if len(batch.solvent.shape) == 1:
        batch_size = torch.max(batch.batch).item() + 1 if hasattr(batch, 'batch') else 1
        if batch.solvent.numel() == batch_size * 4:
            batch.solvent = batch.solvent.view(batch_size, 4)
        else:
            try:
                batch.solvent = batch.solvent.view(-1, 4)
            except:
                pass
    
    if len(batch.y.shape) == 1:
        batch_size = torch.max(batch.batch).item() + 1 if hasattr(batch, 'batch') else 1
        if batch.y.numel() == batch_size * 3:
            batch.y = batch.y.view(batch_size, 3)
    
    return batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
