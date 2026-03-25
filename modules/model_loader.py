import torch
import timm

def load_model():

    model = timm.create_model("tf_efficientnet_b4", pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    model.load_state_dict(torch.load("models/deepfake_model.pth", map_location="cpu"))

    model.eval()

    return model