import torch
import os

def save_model(model=None, epoch=None, save_path_dir=None):
    if os.path.exists(save_path_dir):
        weights_name = 'Epoch-' + str(epoch) + '.pth'
        weights_path = os.path.join(save_path_dir, weights_name)
        torch.save(model.state_dict(), weights_path)
    else:
        os.mkdir(save_path_dir)
        weights_name = 'Epoch-' + str(epoch) + '.pth'
        weights_path = os.path.join(save_path_dir, weights_name)
        torch.save(model.state_dict(), weights_path)
