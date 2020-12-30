import torch

class LossStore:
    def __init__(self, loss_weight_dict):
        self.loss_weight_dict = loss_weight_dict
        self.current_loss_dict = {}

    def total_loss(self):
        """
        Scale losses by their weights and sum. Then clear current_loss_dict
        """
        
        scaled_losses = {k:v*self.loss_weight_dict[k] for k,v in self.current_loss_dict.items()}
        total_loss = torch.stack(tuple(scaled_losses.values())).sum()
        self.current_loss_dict = {}
        return total_loss

    def update_loss(self, name, value):
        assert name in self.loss_weight_dict.keys(), f"No weight for loss {name}"
        self.current_loss_dict[name] = value
