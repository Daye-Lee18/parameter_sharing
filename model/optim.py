import torch.nn as nn 

class InverseSqrtScheduler:
    def __init__(self, optimizer, warmup_updates, warmup_init_lr, peak_lr):
        self.optimizer = optimizer
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.peak_lr = peak_lr
        self.update_step = 0

    def step(self):
        self.update_step += 1
        if self.update_step <= self.warmup_updates:
            lr = self.warmup_init_lr + (self.peak_lr - self.warmup_init_lr) * self.update_step / self.warmup_updates
        else:
            lr = self.peak_lr * (self.update_step ** -0.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'warmup_updates': self.warmup_updates,
            'warmup_init_lr': self.warmup_init_lr,
            'peak_lr': self.peak_lr,
            'update_step': self.update_step
        }

    def load_state_dict(self, state_dict):
        self.warmup_updates = state_dict['warmup_updates']
        self.warmup_init_lr = state_dict['warmup_init_lr']
        self.peak_lr = state_dict['peak_lr']
        self.update_step = state_dict['update_step']
        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = pred.data.clone()
        true_dist.fill_(self.smoothing / (pred.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(pred, true_dist)

