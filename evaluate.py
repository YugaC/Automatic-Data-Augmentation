# evaluate_module.py
import torch
from train_module import initialize_network
from sklearn.metrics import jaccard_score

def load_model(path):
    model = initialize_network()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def evaluate_model(model, test_loader):
    dice_scores = []
    for data, target in test_loader:
        output = model(data)
        output = output.argmax(dim=1, keepdim=True)
        dice_score = jaccard_score(target.numpy(), output.view_as(target).numpy(), average='macro')
        dice_scores.append(dice_score)
    return dice_scores
