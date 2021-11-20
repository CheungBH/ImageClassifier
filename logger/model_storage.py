import torch
import os


class ModelSaver:
    def __init__(self, folder_path):
        self.folder = folder_path
        
    def update(self, model, epoch, best_metrics=()):
        self.save_latest(model)
        if epoch > 0:
            self.save_current_epoch(model, epoch)
        for metric in best_metrics:
            self.save_best_metric(model, metric)

    def save_latest(self, model):
        torch.save(model.state_dict(), os.path.join(self.folder, "latest.pth"))
        
    def save_current_epoch(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.folder, "{}.pth".format(epoch)))
        
    def save_best_metric(self, model, metric):
        torch.save(model.state_dict(), os.path.join(self.folder, "best_{}.pth".format(metric)))



