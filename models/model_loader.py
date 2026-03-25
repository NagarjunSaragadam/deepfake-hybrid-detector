import torch

def load_model():

    model = torch.load("models/deepfake_model.pth", map_location="cpu")
    model.eval()

    return model